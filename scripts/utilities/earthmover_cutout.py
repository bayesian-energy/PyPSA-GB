"""Build an atlite ERA5 weather cutout from the public Earthmover/Arraylake ERA5 dataset.

This is a fast alternative to the CDS ERA5 download (hours) and the Zenodo pre-built cutouts
(~700 MB/year): it reads only the GB box for the requested year directly from object storage and
lets atlite compute the cutout, producing a `uk-<year>.nc` interchangeable with the other tiers.

Source: the public Arraylake repo ``earthmover-public/era5`` (ERA5 hourly, 0.25°, 1940-present).
It is free but requires a (free) Arraylake account — either run ``arraylake auth login`` once, or
set the ``ARRAYLAKE_TOKEN`` environment variable. Optional dependencies (not in the base install):

    pip install arraylake "zarr>=3" icechunk pcodec numcodecs

How it works: atlite's ``get_data_<feature>`` functions call
``atlite.datasets.era5.retrieve_data(variable=[CDS names], ...)`` and expect raw ERA5 short-name
variables back. The Earthmover ``single/temporal`` group holds exactly those variables, so we swap
``retrieve_data`` for one that slices the Earthmover store — atlite's feature maths and the on-disk
cutout schema are reused unchanged.
"""
from __future__ import annotations

import itertools
import logging
import os

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

EARTHMOVER_REPO = "earthmover-public/era5"
EARTHMOVER_GROUP = "single/temporal"  # surface (single-level) fields, time-contiguous chunking

# CDS long name (what atlite requests) -> ERA5 short name (what the store holds).
_CDS_TO_SHORT = {
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "100m_u_component_of_wind": "u100",
    "100m_v_component_of_wind": "v100",
    "forecast_surface_roughness": "fsr",
    "surface_net_solar_radiation": "ssr",
    "surface_solar_radiation_downwards": "ssrd",
    "toa_incident_solar_radiation": "tisr",
    "total_sky_direct_solar_radiation_at_surface": "fdir",
    "2m_temperature": "t2m",
    "soil_temperature_level_4": "stl4",
    "2m_dewpoint_temperature": "d2m",
    "runoff": "ro",
    "geopotential": "z",
}
# atlite reads `.attrs["units"]`; backfill any the store omits.
_DEFAULT_UNITS = {
    "u10": "m s**-1", "v10": "m s**-1", "u100": "m s**-1", "v100": "m s**-1",
    "fsr": "m", "ssr": "J m**-2", "ssrd": "J m**-2", "tisr": "J m**-2",
    "fdir": "J m**-2", "t2m": "K", "stl4": "K", "d2m": "K", "ro": "m", "z": "m**2 s**-2",
}

_STORE_CACHE: dict[str, xr.Dataset] = {}


def _client():
    """Arraylake client. Uses ARRAYLAKE_TOKEN if set, else the cached `arraylake auth login`."""
    from arraylake import Client

    token = os.environ.get("ARRAYLAKE_TOKEN")
    return Client(token=token) if token else Client()


def _open_store(repo: str = EARTHMOVER_REPO) -> xr.Dataset:
    """Open the Earthmover ERA5 group, longitudes normalised to [-180, 180)."""
    if repo not in _STORE_CACHE:
        session = _client().get_repo(repo).readonly_session("main")
        ds = xr.open_zarr(session.store, group=EARTHMOVER_GROUP, consolidated=False)
        lon = ((ds["longitude"] + 180) % 360) - 180  # store is 0..360; atlite uses -180..180
        _STORE_CACHE[repo] = ds.assign_coords(longitude=lon).sortby("longitude")
    return _STORE_CACHE[repo]


def _requested_times(updates: dict) -> pd.DatetimeIndex:
    """The exact hours atlite asked for, from its year/month/day/time lists."""
    years = np.atleast_1d(updates["year"])
    months = np.atleast_1d(updates["month"])
    days = np.atleast_1d(updates.get("day", [f"{d:02d}" for d in range(1, 32)]))
    hours = np.atleast_1d(updates.get("time", [f"{h:02d}:00" for h in range(24)]))
    stamps = [f"{y}-{m}-{d} {hh}" for y, m, d, hh in itertools.product(years, months, days, hours)]
    idx = pd.to_datetime(pd.Series(stamps), errors="coerce").dropna()
    return pd.DatetimeIndex(idx.unique()).sort_values()


def _retrieve_data_earthmover(product=None, chunks=None, tmpdir=None, lock=None, **updates):
    """Drop-in replacement for ``atlite.datasets.era5.retrieve_data`` sourcing from Earthmover."""
    ds = _open_store()
    variables = np.atleast_1d(updates["variable"]).tolist()
    short = [_CDS_TO_SHORT.get(v, v) for v in variables]
    missing = [v for v, s in zip(variables, short) if s not in ds]
    if missing:
        raise KeyError(f"Earthmover ERA5 store lacks variables: {missing}")

    out = ds[short]
    if "area" in updates:  # area = [North, West, South, East]; store latitude is descending
        n, w, s, e = updates["area"]
        out = out.sel(latitude=slice(n, s), longitude=slice(w, e))

    want = _requested_times(updates)
    out = out.sel(valid_time=out.indexes["valid_time"].intersection(want))

    for s in short:
        out[s].attrs.setdefault("units", _DEFAULT_UNITS.get(s, ""))
    # Uniform spatial chunks so atlite's concat/write doesn't hit "inconsistent chunks along y".
    return out.chunk({"latitude": -1, "longitude": -1})


def _patch_atlite() -> None:
    import atlite.datasets.era5 as era5

    era5.retrieve_data = _retrieve_data_earthmover


def build_earthmover_cutout(year, output_path, bounds=None, features=("wind", "influx", "temperature"),
                            dx=0.25, dy=0.25):
    """Build ``uk-<year>.nc`` for the GB box from the Earthmover ERA5 store.

    Parameters
    ----------
    year : int
    output_path : str or Path
    bounds : dict or None
        {"north", "south", "west", "east"}; defaults to the GB box.
    features : sequence of str
        atlite features to prepare. wind + influx + temperature cover GB wind/solar
        (cutout.wind()/cutout.pv()); runoff/height are not needed and not in the store.
    """
    import atlite
    import dask

    b = bounds or {"north": 61.0, "south": 49.5, "west": -11.0, "east": 2.5}
    _patch_atlite()
    cutout = atlite.Cutout(
        path=str(output_path),
        module="era5",
        x=slice(b["west"], b["east"]),
        y=slice(b["south"], b["north"]),
        time=str(year),
        dx=dx,
        dy=dy,
    )
    logger.info(f"Building Earthmover cutout {output_path} ({year}) grid={dict(cutout.coords.sizes)}")
    # atlite runs get_data under dask; arraylake's sync() can't run from a dask worker, so use the
    # synchronous scheduler. monthly_requests splits the read per month to cap peak memory.
    with dask.config.set(scheduler="synchronous"):
        cutout.prepare(features=list(features), monthly_requests=True)
    logger.info(f"  Earthmover cutout complete: {output_path}")
    return cutout
