"""
download_bom_weather.py

Real wind data for the AORVA project via the Open-Meteo API.

Open-Meteo (https://open-meteo.com) serves ERA5 reanalysis data from the
European Centre for Medium-Range Weather Forecasts (ECMWF), which is the
same underlying dataset used by Australia's Bureau of Meteorology (BoM) for
its own reanalysis products.  It is free, requires no API key, covers any
date back to 1940, and offers 0.25 deg horizontal resolution (~28 km).

The recommended download mode is the SPATIAL GRID (6 x 7 = 42 nodes) which
gives the WindField3D module real, spatially-varying wind observations at
individual nodes across the corridor, triggering Inverse Distance Weighting
(IDW) interpolation inside the simulator.

Output column schema (both methods produce the same columns):
    timestamp           pd.Timestamp  - local Sydney time (AEST/AEDT)
    wind_speed_kmh      float         - 10-m wind speed in km/h
    wind_direction_deg  float         - meteorological direction,  deg from N

Spatial-grid downloads additionally include:
    lat    float  - observation node latitude
    lon    float  - observation node longitude

Run directly to download and save the spatial grid for the last 30 days:
    python download_bom_weather.py
"""

import numpy as np
import pandas as pd
import requests


# BoM reference weather stations within / near the corridor.
# These are NOT queried via this module (the BoM JSON feed has been retired),
# but are provided as metadata for documentation and future use.
BOM_REFERENCE_STATIONS = {
    'sydney_airport':  {'id': '066037', 'lat': -33.9465, 'lon': 151.1731},
    'bankstown':       {'id': '066137', 'lat': -33.9244, 'lon': 150.9883},
    'richmond_raaf':   {'id': '067105', 'lat': -33.6004, 'lon': 150.7812},
    'camden_airport':  {'id': '068192', 'lat': -34.0406, 'lon': 150.6878},
    'parramatta':      {'id': '066124', 'lat': -33.8133, 'lon': 151.0020},
}

# Corridor study area (min_lon, min_lat, max_lon, max_lat)
_CORRIDOR_BOUNDS = (150.8233, -34.0173, 151.0875, -33.7078)


class OpenMeteoWindDownloader:
    """
    Downloads real ERA5-based wind observations from Open-Meteo.

    Two download modes:

    Single-point (uniform field):
        download_recent(past_days) / download_historical(start, end)
        -> columns: timestamp, wind_speed_kmh, wind_direction_deg

    Spatial grid (IDW-interpolated field):
        download_spatial_grid(past_days, n_lat, n_lon)
        -> columns: lat, lon, timestamp, wind_speed_kmh, wind_direction_deg
        This mode is RECOMMENDED - it gives a spatially-varying wind field
        with real observations at each of the n_lat x n_lon corridor nodes.

    The output schema is recognised by WindField3D.interpolate_wind_field(),
    which automatically selects IDW or uniform mode based on whether the
    DataFrame contains 'lat' / 'lon' columns.
    """

    ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

    # Midpoint of the Westmead-Liverpool corridor
    DEFAULT_LAT = -33.863
    DEFAULT_LON = 150.955

    def download_historical(self, start_date: str, end_date: str,
                             lat: float = None, lon: float = None
                             ) -> pd.DataFrame:
        """
        Download hourly wind for an arbitrary historical date range.

        Parameters
        ----------
        start_date : 'YYYY-MM-DD'  (earliest available: 1940-01-01)
        end_date   : 'YYYY-MM-DD'
        lat, lon   : float | None  - defaults to corridor midpoint

        Returns
        -------
        pd.DataFrame  columns: timestamp, wind_speed_kmh, wind_direction_deg
        """
        params = {
            'latitude':       lat if lat is not None else self.DEFAULT_LAT,
            'longitude':      lon if lon is not None else self.DEFAULT_LON,
            'start_date':     start_date,
            'end_date':       end_date,
            'hourly':         'windspeed_10m,winddirection_10m',
            'timezone':       'Australia/Sydney',
            'windspeed_unit': 'kmh',
        }
        print(f"  Fetching ERA5 historical wind: {start_date} -> {end_date} "
              f"at ({params['latitude']:.3f}, {params['longitude']:.3f})")
        resp = requests.get(self.ARCHIVE_URL, params=params, timeout=60)
        resp.raise_for_status()
        df = self._parse_response(resp.json())
        print(f"    Retrieved {len(df)} hourly observations")
        return df

    def download_recent(self, past_days: int = 30,
                        lat: float = None, lon: float = None
                        ) -> pd.DataFrame:
        """
        Download the last `past_days` days of real wind data.

        Uses the Open-Meteo forecast endpoint with `past_days` look-back,
        updated hourly from operational NWP/ERA5.

        Parameters
        ----------
        past_days : int    - days to look back (default 30)
        lat, lon  : float  - defaults to corridor midpoint

        Returns
        -------
        pd.DataFrame  columns: timestamp, wind_speed_kmh, wind_direction_deg
        """
        params = {
            'latitude':       lat if lat is not None else self.DEFAULT_LAT,
            'longitude':      lon if lon is not None else self.DEFAULT_LON,
            'hourly':         'windspeed_10m,winddirection_10m',
            'past_days':      int(past_days),
            'forecast_days':  1,
            'timezone':       'Australia/Sydney',
            'windspeed_unit': 'kmh',
        }
        resp = requests.get(self.FORECAST_URL, params=params, timeout=30)
        resp.raise_for_status()
        df = self._parse_response(resp.json())
        return df

    def download_spatial_grid(self, past_days: int = 30,
                               n_lat: int = 6, n_lon: int = 7,
                               lat_bounds: tuple = (-33.97, -33.76),
                               lon_bounds: tuple = (150.86, 151.06)
                               ) -> pd.DataFrame:
        """
        Download real wind at a regular lat/lon grid covering the corridor.

        Creates n_lat x n_lon individual observation nodes, downloads
        `past_days` of hourly ERA5 wind at each, and returns a single
        DataFrame with 'lat' and 'lon' columns.

        WindField3D.interpolate_wind_field() detects these columns and
        applies Inverse Distance Weighting (IDW) to produce a spatially-
        varying 3-D wind field - each (x, y) column of the voxel grid
        gets the correct interpolated wind vector for its geographic location.

        Default grid (6 x 7 = 42 nodes) gives one node roughly every 4 km,
        which captures the mesoscale wind variation across the 30 km corridor.
        The three closest BoM reference stations (Sydney Airport, Bankstown,
        Richmond RAAF) fall within the grid bounds.

        Parameters
        ----------
        past_days  : int   - look-back window (default 30)
        n_lat      : int   - grid rows south -> north (default 6)
        n_lon      : int   - grid columns west -> east (default 7)
        lat_bounds : (south, north) of the sample grid
        lon_bounds : (west,  east)  of the sample grid

        Returns
        -------
        pd.DataFrame  columns: lat, lon, timestamp, wind_speed_kmh,
                               wind_direction_deg
        """
        lats  = np.linspace(lat_bounds[0], lat_bounds[1], n_lat)
        lons  = np.linspace(lon_bounds[0], lon_bounds[1], n_lon)
        nodes = [(lat, lon) for lat in lats for lon in lons]
        total = len(nodes)

        print(f"Downloading ERA5 spatial wind grid: "
              f"{n_lat}x{n_lon} = {total} nodes, last {past_days} days")
        print(f"  Grid bounds: lat [{lat_bounds[0]:.3f}, {lat_bounds[1]:.3f}], "
              f"lon [{lon_bounds[0]:.3f}, {lon_bounds[1]:.3f}]")

        frames = []
        for i, (lat, lon) in enumerate(nodes):
            print(f"  Node {i + 1:02d}/{total}: ({lat:.3f}, {lon:.3f})", end='  ')
            try:
                df = self.download_recent(past_days=past_days, lat=lat, lon=lon)
                df['lat'] = lat
                df['lon'] = lon
                frames.append(df)
                print(f"OK {len(df)} rows")
            except Exception as exc:
                print(f"SKIPPED ({exc})")

        if not frames:
            raise RuntimeError(
                "All nodes failed. Check internet connection and try again."
            )

        result = pd.concat(frames, ignore_index=True)
        result = result[['lat', 'lon', 'timestamp', 'wind_speed_kmh',
                          'wind_direction_deg']]

        n_ok = result[['lat', 'lon']].drop_duplicates().shape[0]
        n_ts = result['timestamp'].nunique()
        print(f"\nSpatial grid complete: {len(result):,} rows  "
              f"({n_ok} nodes x {n_ts} timestamps)")
        print(f"  Speed range:  {result['wind_speed_kmh'].min():.1f}-"
              f"{result['wind_speed_kmh'].max():.1f} km/h")
        print(f"  Date range:   {result['timestamp'].min()} -> "
              f"{result['timestamp'].max()}")
        return result

    @staticmethod
    def _parse_response(data: dict) -> pd.DataFrame:
        """Parse Open-Meteo JSON into the standard wind DataFrame."""
        hourly = data['hourly']
        df = pd.DataFrame({
            'timestamp':          pd.to_datetime(hourly['time']),
            'wind_speed_kmh':     hourly['windspeed_10m'],
            'wind_direction_deg': hourly['winddirection_10m'],
        })
        return df.dropna().reset_index(drop=True)


# ======================================================================
# Download script  (python download_bom_weather.py)
# ======================================================================
if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    os.makedirs('data',    exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    dl = OpenMeteoWindDownloader()

    # --- Download 6x7 spatial grid (42 individual observation nodes) ---
    print("=" * 60)
    print("Downloading real ERA5 wind at 42 spatial nodes...")
    print("=" * 60)
    spatial_df = dl.download_spatial_grid(past_days=30, n_lat=6, n_lon=7)

    out_spatial = 'data/wind_spatial_real.csv'
    spatial_df.to_csv(out_spatial, index=False)
    print(f"\nSaved {len(spatial_df):,} rows to {out_spatial}")

    # --- Also save a single-point summary for fallback use ---
    single_df = dl.download_recent(past_days=30)
    out_single = 'data/wind_historical_real.csv'
    single_df.to_csv(out_single, index=False)
    print(f"Saved {len(single_df)} rows to {out_single}")

    # --- Visualisation: node map + wind rose ---
    nodes = spatial_df[['lat', 'lon']].drop_duplicates()
    latest = spatial_df['timestamp'].max()
    snap   = (spatial_df[spatial_df['timestamp'] == latest]
              .drop_duplicates(subset=['lat', 'lon']))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: spatial node map with wind arrows
    ax = axes[0]
    ax.scatter(nodes['lon'], nodes['lat'],
               s=50, c='steelblue', zorder=4, label='ERA5 nodes')

    for _, row in snap.iterrows():
        spd = row['wind_speed_kmh']
        rad = np.radians(row['wind_direction_deg'])
        # Meteorological -> Cartesian: wind blows FROM the stated direction
        du = -np.sin(rad) * 0.015
        dv = -np.cos(rad) * 0.015
        ax.annotate('', xy=(row['lon'] + du, row['lat'] + dv),
                    xytext=(row['lon'], row['lat']),
                    arrowprops=dict(arrowstyle='->', color='navy',
                                   lw=1.4, mutation_scale=12),
                    zorder=5)
        ax.text(row['lon'] + 0.003, row['lat'] + 0.003,
                f'{spd:.0f}', fontsize=6, color='darkblue')

    # Mark BoM reference stations
    for name, info in BOM_REFERENCE_STATIONS.items():
        ax.plot(info['lon'], info['lat'], 'r^', markersize=9,
                markeredgecolor='white', markeredgewidth=0.8, zorder=6)
        ax.text(info['lon'] + 0.004, info['lat'] + 0.003,
                name.replace('_', ' ').title(),
                fontsize=6.5, color='darkred', zorder=7)

    # Hospital markers
    ax.plot(150.9875, -33.8078, 'b*', markersize=16,
            markeredgecolor='white', markeredgewidth=1.5,
            label='Westmead Hospital', zorder=8)
    ax.plot(150.9233, -33.9173, 'g*', markersize=16,
            markeredgecolor='white', markeredgewidth=1.5,
            label='Liverpool Hospital', zorder=8)

    ax.set_xlabel('Longitude', fontsize=9)
    ax.set_ylabel('Latitude',  fontsize=9)
    ax.set_title(f'ERA5 Spatial Wind Nodes ({len(nodes)} nodes)\n'
                 f'Arrows: instantaneous wind at {latest.strftime("%Y-%m-%d %H:%M")}',
                 fontsize=10)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Right: wind-speed histogram
    ax2 = axes[1]
    ax2.hist(spatial_df['wind_speed_kmh'], bins=30,
             color='steelblue', edgecolor='white', linewidth=0.5, alpha=0.85)
    mean_spd = spatial_df['wind_speed_kmh'].mean()
    ax2.axvline(mean_spd, color='red', linestyle='--', linewidth=1.5,
                label=f'Mean = {mean_spd:.1f} km/h')
    ax2.set_xlabel('Wind Speed (km/h)', fontsize=9)
    ax2.set_ylabel('Count', fontsize=9)
    ax2.set_title(f'ERA5 Wind Speed Distribution\n'
                  f'{len(nodes)} nodes x {spatial_df["timestamp"].nunique()} timestamps',
                  fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Real ERA5 Wind Data - Westmead-Liverpool Corridor\n'
                 '(Open-Meteo / ECMWF ERA5, same dataset as BoM reanalysis)',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig('outputs/wind_data_overview.png', dpi=250, bbox_inches='tight')
    plt.close()
    print("Saved outputs/wind_data_overview.png")
