"""DWD CDC hourly weather data — download, cache, and join to accident records.

Licence: GeoNutzV (Geodatenlizenz Deutschland — uneingeschränkte Nutzung).
Attribution: "Quelle: Deutscher Wetterdienst (DWD), Climate Data Center (CDC)"

Public API
----------
download_station_list   → DataFrame of all DWD stations with coordinates
find_nearest_station    → (station_id, dist_km) for a single point
download_station_data   → hourly observations for one station + variable
build_weather_features  → enrich the accident DataFrame with DWD columns
"""

from __future__ import annotations

import io
import logging
import re
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree
from tqdm import tqdm

log = logging.getLogger(__name__)

__all__ = [
    "download_station_list",
    "find_nearest_station",
    "download_station_data",
    "build_weather_features",
    "VARIABLES",
]

DWD_BASE = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly"

VARIABLES: dict[str, dict[str, str]] = {
    "TU": {
        "name": "air_temperature",
        "col_raw": "TT_TU",
        "col_out": "dwd_temp_air_2m",
    },
    "RR": {
        "name": "precipitation",
        "col_raw": "R1",
        "col_out": "dwd_precip_mm",
    },
    "VV": {
        "name": "visibility",
        "col_raw": "V_VV",
        "col_out": "dwd_visibility_m",
    },
    "FF": {
        "name": "wind",
        "col_raw": "F",
        "col_out": "dwd_wind_speed_ms",
    },
}

STATION_LIST_URL = f"{DWD_BASE}/air_temperature/recent/TU_Stundenwerte_Beschreibung_Stationen.txt"

_DWD_SENTINELS = {-999.0, -9999.0}


# ---------------------------------------------------------------------------
# Station list
# ---------------------------------------------------------------------------


def download_station_list(
    cache_dir: Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download and parse the DWD station master list.

    Returns a DataFrame with columns:
        station_id (str, zero-padded to 5 digits), lat (float), lon (float),
        elevation_m (int), date_start (Timestamp), date_end (Timestamp),
        name (str), bundesland (str)

    File is cached to cache_dir/station_list.txt.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "station_list.txt"

    if not cache_path.exists() or force_refresh:
        log.info("Fetching DWD station list from %s", STATION_LIST_URL)
        resp = requests.get(STATION_LIST_URL, timeout=30)
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)
        log.info("Cached station list → %s", cache_path)

    raw = cache_path.read_text(encoding="latin-1")
    lines = [ln for ln in raw.splitlines() if ln.strip() and not set(ln.strip()) <= {"-"}]

    rows = []
    for line in lines[1:]:  # skip header row
        parts = line.split(None, 7)
        if len(parts) < 8:
            continue
        rows.append(parts)

    df = pd.DataFrame(
        rows,
        columns=[
            "station_id",
            "date_start",
            "date_end",
            "elevation_m",
            "lat",
            "lon",
            "name",
            "bundesland",
        ],
    )
    df["station_id"] = df["station_id"].str.zfill(5)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["elevation_m"] = pd.to_numeric(df["elevation_m"], errors="coerce")
    df["date_start"] = pd.to_datetime(df["date_start"], format="%Y%m%d", errors="coerce")
    df["date_end"] = pd.to_datetime(df["date_end"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["lat", "lon"])

    if len(df) < 100:
        log.warning("Station list has only %d entries — possible parse error", len(df))

    log.info("Station list: %d entries", len(df))
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Spatial helpers
# ---------------------------------------------------------------------------


def _build_station_tree(stations: pd.DataFrame) -> tuple[cKDTree, np.ndarray]:
    """Build a cKDTree over (lat, lon) in radians for fast nearest-station lookup.

    Euclidean distance in radian-space is accurate to <0.1 % for distances ≤ 30 km.
    """
    coords_rad = np.deg2rad(stations[["lat", "lon"]].values)
    tree = cKDTree(coords_rad)
    return tree, stations["station_id"].values


def find_nearest_station(
    lat: float,
    lon: float,
    stations: pd.DataFrame,
    tree: cKDTree | None = None,
    max_km: float = 30.0,
) -> tuple[str | None, float]:
    """Return (station_id, distance_km) for the nearest DWD station.

    Returns (None, distance_km) if no station is within max_km.
    Pass a pre-built tree when calling in a loop for performance.
    """
    station_ids: np.ndarray
    if tree is None:
        tree, station_ids = _build_station_tree(stations)
    else:
        station_ids = stations["station_id"].values

    query_rad = np.deg2rad([[lat, lon]])
    dist_rad, idx = tree.query(query_rad, k=1)
    dist_km = float(dist_rad[0]) * 6371.0

    if dist_km > max_km:
        return None, dist_km
    return str(station_ids[idx[0]]), dist_km


# ---------------------------------------------------------------------------
# Per-station download
# ---------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _fetch_dir_index(var_name: str, variable_code: str, period: str) -> dict[str, str]:
    """Fetch DWD directory listing once and return {station_id: zip_filename}.

    Cached per (var_name, variable_code, period) so the HTTP request is made
    only once per enrichment run regardless of how many stations are queried.
    """
    dir_url = f"{DWD_BASE}/{var_name}/{period}/"
    try:
        resp = requests.get(dir_url, timeout=30)
        if resp.status_code != 200:
            log.warning("Directory listing failed (%d): %s", resp.status_code, dir_url)
            return {}
        pattern = rf'(stundenwerte_{variable_code}_(\d{{5}})_[^"<\s]+\.zip)'
        return {sid: fname for fname, sid in re.findall(pattern, resp.text)}
    except Exception as exc:
        log.warning("Failed to fetch directory listing %s: %s", dir_url, exc)
        return {}


def download_station_data(
    station_id: str,
    variable_code: str,
    start_year: int,
    end_year: int,
    cache_dir: Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download and parse hourly DWD data for one station and one variable.

    variable_code: one of 'TU', 'RR', 'VV', 'FF'

    Returns DataFrame with columns: [timestamp (UTC, tz-aware), <col_out>]
    Missing/invalid measurements (-999, -9999) are replaced with NaN.
    Returns an empty DataFrame (with the correct columns) on any download failure.
    """
    if variable_code not in VARIABLES:
        raise KeyError(f"Unknown variable code: {variable_code!r}. Use one of {list(VARIABLES)}")

    var = VARIABLES[variable_code]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sid = str(station_id).zfill(5)
    cache_path = cache_dir / f"{variable_code}_{sid}_{start_year}_{end_year}.parquet"

    if cache_path.exists() and not force_refresh:
        return pd.read_parquet(cache_path)

    frames: list[pd.DataFrame] = []
    var_name = var["name"]

    for period in ("historical", "recent"):
        # Recent files always use _akt suffix.
        # Historical filenames carry a station-specific date range and must be
        # discovered from the directory listing (fetched once, cached in memory).
        if period == "recent":
            candidate_urls = [
                f"{DWD_BASE}/{var_name}/{period}/stundenwerte_{variable_code}_{sid}_akt.zip",
            ]
        else:
            dir_index = _fetch_dir_index(var_name, variable_code, period)
            hist_fname = dir_index.get(sid)
            if not hist_fname:
                continue
            candidate_urls = [f"{DWD_BASE}/{var_name}/{period}/{hist_fname}"]

        for url in candidate_urls:
            try:
                resp = requests.get(url, timeout=60)
                if resp.status_code == 404:
                    continue
                resp.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                    data_files = [n for n in zf.namelist() if n.startswith("produkt_")]
                    for zip_fname in data_files:
                        with zf.open(zip_fname) as fh:
                            chunk = pd.read_csv(fh, sep=";", encoding="latin-1", dtype=str)
                            frames.append(chunk)
                break  # found a working URL for this period
            except requests.HTTPError as exc:
                log.debug("HTTP error %s for %s: %s", exc.response.status_code, url, exc)
            except Exception as exc:
                log.debug("Error fetching %s: %s", url, exc)

    col_out = var["col_out"]
    empty = pd.DataFrame(columns=["timestamp", col_out])

    if not frames:
        log.warning("No data found for station %s variable %s", sid, variable_code)
        return empty

    raw = pd.concat(frames, ignore_index=True)
    raw.columns = raw.columns.str.strip()

    raw["timestamp"] = pd.to_datetime(
        raw["MESS_DATUM"].str.strip(),
        format="%Y%m%d%H",
        errors="coerce",
        utc=True,
    )

    # DWD column headers sometimes have leading/trailing spaces — find by stripped name
    col_raw_stripped = var["col_raw"].strip()
    matching = [c for c in raw.columns if c.strip() == col_raw_stripped]
    if not matching:
        log.warning(
            "Expected column %r not found in %s data for station %s. Columns: %s",
            col_raw_stripped,
            variable_code,
            sid,
            list(raw.columns),
        )
        return empty

    raw[col_out] = pd.to_numeric(raw[matching[0]], errors="coerce")
    raw.loc[raw[col_out].isin(_DWD_SENTINELS), col_out] = np.nan

    result = (
        raw[["timestamp", col_out]]
        .dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"])
    )
    result = result[
        (result["timestamp"].dt.year >= start_year) & (result["timestamp"].dt.year <= end_year)
    ].reset_index(drop=True)

    result.to_parquet(cache_path, index=False)
    log.info("Cached %s station %s → %s (%d rows)", variable_code, sid, cache_path, len(result))
    return result


# ---------------------------------------------------------------------------
# Full enrichment
# ---------------------------------------------------------------------------


def build_weather_features(
    accidents_df: pd.DataFrame,
    raw_cache_dir: Path,
    interim_cache_dir: Path,
    max_km: float = 30.0,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Join DWD hourly weather observations to the accident DataFrame.

    Adds columns to accidents_df:
        dwd_station_id        nearest DWD station (str, NaN if > max_km away)
        dwd_station_dist_km   distance to nearest station (float)
        dwd_temp_air_2m       air temperature in °C (float)
        dwd_precip_mm         precipitation in mm (float)
        dwd_visibility_m      visibility in metres (float)
        dwd_wind_speed_ms     wind speed in m/s (float)

    Join granularity: (station_id, UJAHR, UMONAT, USTUNDE) — averaged over all
    days in the same month-hour bucket because accidents.parquet lacks UTAG.
    This is a known approximation documented in §11 limitation 9 of the Q phase.

    Rows with no station within max_km receive NaN for all weather columns.
    The enriched DataFrame is cached to interim_cache_dir/accidents_with_weather.parquet.
    """
    required = {"LAT", "LON", "UJAHR", "UMONAT", "USTUNDE"}
    missing_cols = required - set(accidents_df.columns)
    if missing_cols:
        raise RuntimeError(
            f"accidents_df is missing required columns: {missing_cols}\n"
            "Ensure you are passing a DataFrame loaded from data/accidents.parquet."
        )

    raw_cache_dir = Path(raw_cache_dir)
    interim_cache_dir = Path(interim_cache_dir)
    interim_cache_dir.mkdir(parents=True, exist_ok=True)

    out_path = interim_cache_dir / "accidents_with_weather.parquet"
    if out_path.exists() and not force_refresh:
        log.info("Using cached enriched DataFrame at %s", out_path)
        return pd.read_parquet(out_path)

    # Compute year range up front — needed to filter the station list.
    year_min = int(accidents_df["UJAHR"].min())
    year_max = int(accidents_df["UJAHR"].max())

    # --- Station list + spatial tree ---
    stations = download_station_list(raw_cache_dir, force_refresh=force_refresh)
    # Drop stations decommissioned before the accident period to avoid
    # assigning accidents to stations that have no data for 2016–present.
    stations = stations[
        stations["date_end"].isna() | (stations["date_end"] >= pd.Timestamp(f"{year_min}-01-01"))
    ]
    log.info("Active stations (date_end ≥ %d): %d", year_min, len(stations))
    tree, station_ids_arr = _build_station_tree(stations)

    # --- Vectorised nearest-station assignment (all rows in one query) ---
    coords_rad = np.deg2rad(accidents_df[["LAT", "LON"]].values)
    dist_rad, idx = tree.query(coords_rad, k=1, workers=-1)
    dist_km = dist_rad * 6371.0

    df = accidents_df.copy()
    df["dwd_station_id"] = np.where(dist_km <= max_km, station_ids_arr[idx], np.nan)
    df["dwd_station_dist_km"] = dist_km.astype(float)

    # --- Unique stations to fetch ---
    unique_stations = df["dwd_station_id"].dropna().unique()
    log.info(
        "Fetching weather for %d stations, years %d–%d",
        len(unique_stations),
        year_min,
        year_max,
    )

    # --- Download all variables for each unique station ---
    weather_frames: dict[str, list[pd.DataFrame]] = {code: [] for code in VARIABLES}
    for sid in tqdm(unique_stations, desc="DWD stations"):
        for code in VARIABLES:
            try:
                var_df = download_station_data(
                    sid,
                    code,
                    year_min,
                    year_max,
                    cache_dir=raw_cache_dir / code,
                    force_refresh=force_refresh,
                )
                if not var_df.empty:
                    var_df = var_df.copy()
                    var_df["dwd_station_id"] = sid
                    weather_frames[code].append(var_df)
            except Exception as exc:
                log.warning("Failed %s for station %s: %s", code, sid, exc)

    # --- Build join key: (station_id, year, month, hour-of-day) ---
    # accidents.parquet lacks UTAG (day of month) → average over all days in the bucket.
    df["_jk"] = (
        df["UJAHR"].astype(str)
        + "_"
        + df["UMONAT"].astype(str).str.zfill(2)
        + "_"
        + df["USTUNDE"].astype(str).str.zfill(2)
    )

    for code, var in VARIABLES.items():
        col_out = var["col_out"]
        if not weather_frames[code]:
            df[col_out] = np.nan
            continue

        weather = pd.concat(weather_frames[code], ignore_index=True)
        weather["_jk"] = (
            weather["timestamp"].dt.year.astype(str)
            + "_"
            + weather["timestamp"].dt.month.astype(str).str.zfill(2)
            + "_"
            + weather["timestamp"].dt.hour.astype(str).str.zfill(2)
        )
        agg = weather.groupby(["dwd_station_id", "_jk"])[col_out].mean().reset_index()
        df = df.merge(agg, on=["dwd_station_id", "_jk"], how="left")

    df = df.drop(columns=["_jk"])
    df.to_parquet(out_path, index=False)
    log.info("Saved enriched DataFrame → %s (%d rows)", out_path, len(df))
    return df
