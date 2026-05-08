# download_bom_weather.py

import os
import requests
import pandas as pd
import numpy as np


"""
BoM Weather Data Downloader

This script downloads real wind observation data from BoM JSON feeds.
The output is formatted so it can later replace the synthetic wind data.

Main output:
- data/bom_weather_current.csv
- data/wind_historical_bom.csv
"""


class BOMWeatherDownloader:
    def __init__(self):
        self.base_url = "https://www.bom.gov.au/fwo"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Referer": "https://www.bom.gov.au/products/IDN60901/",
        }

        # Stations around the Westmead-Liverpool corridor.
        # product_code is used in the BoM JSON URL.
        # station_id is the official BoM station ID.
        self.stations = {
            "parramatta_north": {
                "product_code": "94764",
                "station_id": "066124"
            },
            "bankstown_airport": {
                "product_code": "94765",
                "station_id": "066137"
            },
            "holsworthy_aerodrome": {
                "product_code": "95761",
                "station_id": "066161"
            },
            "horsley_park": {
                "product_code": "94760",
                "station_id": "067119"
            }
        }

        self.direction_to_degrees = {
            "N": 0,
            "NNE": 22.5,
            "NE": 45,
            "ENE": 67.5,
            "E": 90,
            "ESE": 112.5,
            "SE": 135,
            "SSE": 157.5,
            "S": 180,
            "SSW": 202.5,
            "SW": 225,
            "WSW": 247.5,
            "W": 270,
            "WNW": 292.5,
            "NW": 315,
            "NNW": 337.5,
            "CALM": np.nan
        }

    def download_observations(self, station_name, product_code, station_id):
        """Download latest observations from one BoM station."""

        url = f"{self.base_url}/IDN60901/IDN60901.{product_code}.json"

        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            observations = data["observations"]["data"]
            df = pd.DataFrame(observations)

            if df.empty:
                print(f"No observations found for {station_name}")
                return pd.DataFrame()

            df["timestamp"] = pd.to_datetime(
                df["local_date_time_full"],
                format="%Y%m%d%H%M%S",
                errors="coerce"
            )

            df["wind_speed_kmh"] = pd.to_numeric(df.get("wind_spd_kmh"), errors="coerce")
            df["gust_speed_kmh"] = pd.to_numeric(df.get("gust_kmh"), errors="coerce")
            df["temperature_c"] = pd.to_numeric(df.get("air_temp"), errors="coerce")

            df["wind_direction"] = df.get("wind_dir")
            df["wind_direction_deg"] = df["wind_direction"].map(self.direction_to_degrees)

            df["station"] = station_name
            df["station_id"] = station_id
            df["product_code"] = product_code
            df["hour"] = df["timestamp"].dt.hour

            output_columns = [
                "timestamp",
                "station",
                "station_id",
                "product_code",
                "wind_speed_kmh",
                "wind_direction",
                "wind_direction_deg",
                "gust_speed_kmh",
                "temperature_c",
                "hour"
            ]

            df = df[output_columns].copy()

            print(f"Downloaded {len(df)} observations from {station_name}")
            return df

        except Exception as e:
            print(f"Error downloading {station_name}: {e}")
            return pd.DataFrame()

    def download_all_stations(self):
        """Download observations from all selected stations."""

        all_data = []

        for station_name, station_info in self.stations.items():
            df = self.download_observations(
                station_name=station_name,
                product_code=station_info["product_code"],
                station_id=station_info["station_id"]
            )

            if not df.empty:
                all_data.append(df)

        if not all_data:
            return pd.DataFrame()

        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.sort_values("timestamp")
        return combined


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    downloader = BOMWeatherDownloader()
    weather_df = downloader.download_all_stations()

    if not weather_df.empty:
        weather_df.to_csv("data/bom_weather_current.csv", index=False)

        # Keep only rows that can be used by the 3D wind field model.
        wind_model_df = weather_df.dropna(
            subset=["wind_speed_kmh", "wind_direction_deg"]
        ).copy()

        wind_model_df = wind_model_df.sort_values("timestamp")

        # This file is formatted to replace the old synthetic wind CSV.
        wind_model_df.to_csv("data/wind_historical_bom.csv", index=False)

        print("\n=== Weather Data Summary ===")
        print(weather_df.head(10))

        print("\n=== Wind Statistics ===")
        print(wind_model_df[["wind_speed_kmh", "gust_speed_kmh", "wind_direction_deg"]].describe())

        print("\nSaved files:")
        print("- data/bom_weather_current.csv")
        print("- data/wind_historical_bom.csv")

        print(f"\nRows saved for wind model: {len(wind_model_df)}")
    else:
        print("No weather data downloaded")