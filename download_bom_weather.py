"""
download_bom_weather.py

Two wind downloaders:

  OpenMeteoWindDownloader  (PRIMARY)
      Real ERA5 reanalysis data via Open-Meteo API.
      No API key required.  Supports:
        - download_recent(past_days)          -> single-point DataFrame
        - download_historical(start, end)     -> single-point DataFrame
        - download_spatial_grid(past_days, n_lat, n_lon)
                                              -> multi-node DataFrame
      Spatial-grid output triggers IDW interpolation in WindField3D,
      giving each voxel its own spatially-varying wind vector.

  BOMWeatherDownloader  (LEGACY / fallback)
      Scrapes live BoM JSON observations from three stations near the
      Westmead-Liverpool corridor. Useful for quick real-time checks but
      has no historical depth and requires a live internet connection.

Column schema shared by both downloaders
-----------------------------------------
    timestamp           datetime64[ns]
    wind_speed_kmh      float   (speed at 10 m reference height)
    wind_direction_deg  float   (meteorological: direction FROM, 0=N)
    gust_kmh            float   (OpenMeteo: hourly; BoM: instantaneous)
    temperature_c       float
    -- spatial grid only --
    lat                 float
    lon                 float
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import requests


# ======================================================================
# OpenMeteoWindDownloader  (PRIMARY)
# ======================================================================

# Study-area centre (midpoint of Westmead–Liverpool corridor)
_CENTRE_LAT = -33.8626
_CENTRE_LON = 150.9554

# Bounding box for spatial grid
_BBOX = dict(
    lat_min=-34.0173, lat_max=-33.7078,
    lon_min=150.8233, lon_max=151.0875,
)


class OpenMeteoWindDownloader:
    """
    Download ERA5 wind data from the Open-Meteo Historical Weather API.

    ERA5 is the same reanalysis dataset used by the Bureau of Meteorology
    for its own products. Free, no API key, ~1-hour resolution.
    """

    BASE_URL        = "https://archive-api.open-meteo.com/v1/archive"
    FORECAST_URL    = "https://api.open-meteo.com/v1/forecast"
    HOURLY_PARAMS   = "wind_speed_10m,wind_direction_10m,wind_gusts_10m,temperature_2m"
    RETRY_DELAY_S   = 2

    # ------------------------------------------------------------------
    # Single-point helpers
    # ------------------------------------------------------------------
    def _fetch_one(self, lat: float, lon: float,
                   start: str, end: str) -> pd.DataFrame:
        """Fetch hourly ERA5 for a single lat/lon, date range YYYY-MM-DD."""
        params = dict(
            latitude=lat, longitude=lon,
            start_date=start, end_date=end,
            hourly=self.HOURLY_PARAMS,
            wind_speed_unit="kmh",
            timezone="Australia/Sydney",
        )
        for attempt in range(3):
            try:
                r = requests.get(self.BASE_URL, params=params, timeout=30)
                r.raise_for_status()
                j = r.json()
                h = j.get("hourly", {})
                df = pd.DataFrame({
                    "timestamp":           pd.to_datetime(h["time"]),
                    "wind_speed_kmh":      h["wind_speed_10m"],
                    "wind_direction_deg":  h["wind_direction_10m"],
                    "gust_kmh":            h["wind_gusts_10m"],
                    "temperature_c":       h["temperature_2m"],
                    "lat": lat,
                    "lon": lon,
                })
                return df.dropna(subset=["wind_speed_kmh"])
            except Exception as e:
                if attempt < 2:
                    time.sleep(self.RETRY_DELAY_S)
                else:
                    print(f"  [warn] fetch failed ({lat:.3f},{lon:.3f}): {e}")
                    return pd.DataFrame()

    def _date_range(self, past_days: int) -> tuple[str, str]:
        end   = date.today() - timedelta(days=1)   # ERA5 has ~1-day lag
        start = end - timedelta(days=past_days - 1)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def download_recent(self, past_days: int = 30) -> pd.DataFrame:
        """Single-point download for the corridor centre, last N days."""
        start, end = self._date_range(past_days)
        print(f"Downloading ERA5 (single-point) {start} -> {end} ...")
        df = self._fetch_one(_CENTRE_LAT, _CENTRE_LON, start, end)
        print(f"  {len(df)} hourly observations")
        return df

    def download_historical(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Single-point download for an explicit date range."""
        print(f"Downloading ERA5 (historical) {start_date} -> {end_date} ...")
        df = self._fetch_one(_CENTRE_LAT, _CENTRE_LON, start_date, end_date)
        print(f"  {len(df)} hourly observations")
        return df

    def download_spatial_grid(self, past_days: int = 30,
                               n_lat: int = 6,
                               n_lon: int = 7) -> pd.DataFrame:
        """
        Download ERA5 for a regular n_lat × n_lon grid covering the study
        area. Returns a DataFrame with lat/lon columns so that WindField3D
        can apply IDW interpolation.

        Parameters
        ----------
        past_days : int
            How many days of recent data to fetch.
        n_lat : int
            Number of rows (south → north).
        n_lon : int
            Number of columns (west → east).
        """
        start, end = self._date_range(past_days)
        lats = np.linspace(_BBOX["lat_min"], _BBOX["lat_max"], n_lat)
        lons = np.linspace(_BBOX["lon_min"], _BBOX["lon_max"], n_lon)

        n_nodes = n_lat * n_lon
        print(f"Downloading ERA5 spatial grid: {n_lat}×{n_lon} = {n_nodes} nodes")
        print(f"Date range: {start} -> {end}")

        frames = []
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                node = i * n_lon + j + 1
                print(f"  Node {node}/{n_nodes}  ({lat:.3f}, {lon:.3f})")
                df = self._fetch_one(lat, lon, start, end)
                if not df.empty:
                    frames.append(df)
                time.sleep(0.15)   # polite rate limiting

        if not frames:
            print("ERROR: no data returned from any node.")
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values(["timestamp", "lat", "lon"])
        print(f"\nSpatial grid complete: {len(combined):,} rows, "
              f"{combined[['lat','lon']].drop_duplicates().shape[0]} nodes")
        return combined


# ======================================================================
# BOMWeatherDownloader  (LEGACY)
# ======================================================================

class BOMWeatherDownloader:
    """
    Scrape live BoM JSON observations from three stations near the
    Westmead–Liverpool corridor.

    Stations
    --------
      Sydney Airport  066037
      Bankstown        066137
      Richmond RAAF    067105

    NOTE: BoM observations cover only the most recent ~72 hours and
    have no historical depth. Use OpenMeteoWindDownloader for training.
    """

    BASE_URL = "http://www.bom.gov.au/fwo"
    STATIONS = {
        "sydney_airport": "066037",
        "bankstown":       "066137",
        "richmond":        "067105",
    }

    def _download_station(self, station_id: str,
                           station_name: str) -> pd.DataFrame:
        url = f"{self.BASE_URL}/IDN60901/IDN60901.{station_id}.json"
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            obs = r.json()["observations"]["data"]
            df = pd.DataFrame(obs)
            df["local_date_time_full"] = pd.to_datetime(
                df["local_date_time_full"], format="%Y%m%d%H%M%S"
            )
            cols = ["local_date_time_full", "wind_spd_kmh",
                    "wind_dir", "gust_kmh", "air_temp", "press_msl"]
            df = df[[c for c in cols if c in df.columns]].copy()
            df["station"]    = station_name
            df["station_id"] = station_id
            # Normalise column names to shared schema
            df = df.rename(columns={
                "local_date_time_full": "timestamp",
                "wind_spd_kmh":         "wind_speed_kmh",
                "wind_dir":             "wind_direction_deg",
                "air_temp":             "temperature_c",
            })
            print(f"  {station_name}: {len(df)} observations")
            return df
        except Exception as e:
            print(f"  [warn] {station_name}: {e}")
            return pd.DataFrame()

    def download_all_stations(self) -> pd.DataFrame:
        frames = []
        for name, sid in self.STATIONS.items():
            df = self._download_station(sid, name)
            if not df.empty:
                frames.append(df)
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            return combined.sort_values("timestamp")
        return pd.DataFrame()
