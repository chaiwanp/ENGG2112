"""
generate_synthetic_wind_data.py

This file previously generated synthetic wind data.
It has now been replaced with a BoM real wind data extractor.

Purpose:
- Download real wind observations from Bureau of Meteorology JSON feeds
- Save the raw current weather observations
- Save a wind-model-ready CSV file for the 3D wind field program

Main outputs:
- data/bom_weather_current.csv
- data/wind_historical_bom.csv
"""

import os
import importlib.util
from pathlib import Path


def load_bom_downloader_class():
    """
    Load BOMWeatherDownloader from the BoM downloader file.

    The project keeps development files with a leading '#',
    so normal Python imports may not work reliably.
    This function loads the file directly by path.
    """

    project_dir = Path(__file__).resolve().parent

    possible_paths = [
        project_dir / "# download_bom_weather.py",
        project_dir / "#download_bom_weather.py",
        project_dir / "download_bom_weather.py",
    ]

    downloader_path = None

    for path in possible_paths:
        if path.exists():
            downloader_path = path
            break

    if downloader_path is None:
        raise FileNotFoundError(
            "Could not find the BoM downloader file. "
            "Expected one of: '# download_bom_weather.py', "
            "'#download_bom_weather.py', or 'download_bom_weather.py'."
        )

    spec = importlib.util.spec_from_file_location(
        "download_bom_weather",
        downloader_path
    )

    bom_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bom_module)

    return bom_module.BOMWeatherDownloader


def main():
    os.makedirs("data", exist_ok=True)

    BOMWeatherDownloader = load_bom_downloader_class()
    downloader = BOMWeatherDownloader()

    weather_df = downloader.download_all_stations()

    if weather_df.empty:
        print("No weather data downloaded.")
        return

    weather_df.to_csv("data/bom_weather_current.csv", index=False)

    wind_model_df = weather_df.dropna(
        subset=["wind_speed_kmh", "wind_direction_deg"]
    ).copy()

    wind_model_df = wind_model_df.sort_values("timestamp")
    wind_model_df.to_csv("data/wind_historical_bom.csv", index=False)

    print("\n=== BoM Weather Data Summary ===")
    print(weather_df.head(10))

    print("\n=== Wind Statistics ===")
    print(
        wind_model_df[
            ["wind_speed_kmh", "gust_speed_kmh", "wind_direction_deg"]
        ].describe()
    )

    print("\nSaved files:")
    print("- data/bom_weather_current.csv")
    print("- data/wind_historical_bom.csv")

    print(f"\nRows saved for wind model: {len(wind_model_df)}")
    print("\nSynthetic wind generation has been replaced by real BoM wind data extraction.")


if __name__ == "__main__":
    main()