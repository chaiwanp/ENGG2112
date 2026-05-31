"""
Step 0 - Download real ERA5 wind data via Open-Meteo.

No prerequisites. Run this BEFORE any other step.

DEFAULT: Downloads wind at a 6x7 = 42 spatial grid of individual observation
nodes covering the Westmead-Liverpool corridor. This triggers Inverse Distance
Weighting (IDW) inside WindField3D, giving each position in the voxel grid
its own spatially-interpolated wind vector from real ERA5 data.

Output (spatial - default):
    data/wind_spatial_real.csv
    Columns: lat, lon, timestamp, wind_speed_kmh, wind_direction_deg

Output (single-point - --no-spatial):
    data/wind_historical_real.csv
    Columns: timestamp, wind_speed_kmh, wind_direction_deg

ERA5 data is the same underlying dataset used by Australia's Bureau of
Meteorology for its reanalysis products.  Free, no API key required.

Usage examples:
    python scripts/00_download_real_wind.py                     # spatial, last 30 days
    python scripts/00_download_real_wind.py --days 90           # spatial, last 90 days
    python scripts/00_download_real_wind.py --start 2024-01-01 --end 2024-12-31
    python scripts/00_download_real_wind.py --no-spatial        # single-point only
    python scripts/00_download_real_wind.py --n-lat 3 --n-lon 4 # smaller grid
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from download_bom_weather import OpenMeteoWindDownloader


def main():
    parser = argparse.ArgumentParser(
        description="Download real ERA5 wind data from Open-Meteo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--days', type=int, default=30,
                        help='Days of recent data to fetch')
    parser.add_argument('--start', type=str, default=None,
                        help='Historical start date YYYY-MM-DD (overrides --days)')
    parser.add_argument('--end', type=str, default=None,
                        help='Historical end date YYYY-MM-DD (requires --start)')

    spatial_group = parser.add_mutually_exclusive_group()
    spatial_group.add_argument('--spatial', dest='spatial', action='store_true',
                                default=True,
                                help='Download 6x7 spatial grid (DEFAULT)')
    spatial_group.add_argument('--no-spatial', dest='spatial', action='store_false',
                                help='Download single-point only')

    parser.add_argument('--n-lat', type=int, default=6,
                        help='Spatial grid rows (south -> north)')
    parser.add_argument('--n-lon', type=int, default=7,
                        help='Spatial grid columns (west -> east)')
    args = parser.parse_args()

    os.makedirs('data', exist_ok=True)
    dl = OpenMeteoWindDownloader()

    print("=" * 60)
    print("AORVA Step 0 - Download real ERA5 wind data")
    print("=" * 60)

    if args.spatial:
        # -- Spatial grid (RECOMMENDED) ------------------------------
        if args.start:
            print("NOTE: historical date range not supported for spatial grid.")
            print("      Defaulting to the last --days days.\n")

        wind_df = dl.download_spatial_grid(
            past_days=args.days,
            n_lat=args.n_lat,
            n_lon=args.n_lon,
        )
        out = 'data/wind_spatial_real.csv'
        wind_df.to_csv(out, index=False)
        n_nodes = wind_df[['lat', 'lon']].drop_duplicates().shape[0]
        n_ts    = wind_df['timestamp'].nunique()
        print(f"\nSaved {len(wind_df):,} rows -> {out}")
        print(f"  Nodes: {n_nodes}  |  Timestamps: {n_ts}")
        print(f"  Speed range: {wind_df['wind_speed_kmh'].min():.1f}-"
              f"{wind_df['wind_speed_kmh'].max():.1f} km/h")
        print(f"  Date range:  {wind_df['timestamp'].min()} -> "
              f"{wind_df['timestamp'].max()}")
        print("\nTo use in training:  AORVAEnv(wind_df_path='data/wind_spatial_real.csv')")
        print("WindField3D will use IDW to interpolate over all", n_nodes, "nodes.")

    elif args.start:
        # -- Historical single-point ----------------------------------
        from datetime import date
        end = args.end or date.today().strftime('%Y-%m-%d')
        wind_df = dl.download_historical(args.start, end)
        out = 'data/wind_historical_real.csv'
        wind_df.to_csv(out, index=False)
        print(f"\nSaved {len(wind_df)} rows -> {out}")
        print(f"  Date range: {wind_df['timestamp'].min()} -> "
              f"{wind_df['timestamp'].max()}")
        print("\nTo use in training:  AORVAEnv(wind_df_path='data/wind_historical_real.csv')")

    else:
        # -- Recent single-point --------------------------------------
        wind_df = dl.download_recent(past_days=args.days)
        out = 'data/wind_historical_real.csv'
        wind_df.to_csv(out, index=False)
        print(f"\nSaved {len(wind_df)} rows -> {out}")
        print(f"  Date range: {wind_df['timestamp'].min()} -> "
              f"{wind_df['timestamp'].max()}")
        print("\nTIP: run with --spatial to get a richer spatially-varying wind field.")

    if wind_df.empty:
        print("\nERROR: No data returned. Check internet connection.")
        sys.exit(1)

    print("\nStep 0 complete.")


if __name__ == "__main__":
    main()

