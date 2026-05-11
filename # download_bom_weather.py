# download_bom_weather.py
import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import numpy as np

"""
BoM Weather Data Sources:
1. Historical wind data from nearby weather stations
2. Real-time observations (for testing)

Nearest stations to route:
- Sydney Airport (Station 066037)
- Bankstown Airport (Station 066137) 
- Richmond RAAF (Station 067105)
"""

class BOMWeatherDownloader:
    def __init__(self):
        self.base_url = "http://www.bom.gov.au/fwo"
        # Stations near Westmead-Liverpool route
        self.stations = {
            'sydney_airport': '066037',
            'bankstown': '066137',
            'richmond': '067105'
        }
        
    def download_observations(self, station_id, station_name):
        """Download latest observations from BoM"""
        url = f"{self.base_url}/IDN60901/IDN60901.{station_id}.json"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Extract observations
            obs = data['observations']['data']
            df = pd.DataFrame(obs)
            
            # Convert to datetime
            df['local_date_time_full'] = pd.to_datetime(
                df['local_date_time_full'], 
                format='%Y%m%d%H%M%S'
            )
            
            # Select relevant columns
            cols = [
                'local_date_time_full',
                'wind_spd_kmh', 
                'wind_dir',
                'gust_kmh',
                'air_temp',
                'press_msl'
            ]
            
            df = df[cols].copy()
            df['station'] = station_name
            df['station_id'] = station_id
            
            print(f"Downloaded {len(df)} observations from {station_name}")
            return df
            
        except Exception as e:
            print(f"Error downloading {station_name}: {e}")
            return pd.DataFrame()
    
    def download_all_stations(self):
        """Download from all stations"""
        all_data = []
        
        for name, station_id in self.stations.items():
            df = self.download_observations(station_id, name)
            if not df.empty:
                all_data.append(df)
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            combined = combined.sort_values('local_date_time_full')
            return combined
        return pd.DataFrame()

# Download weather data
downloader = BOMWeatherDownloader()
weather_df = downloader.download_all_stations()

if not weather_df.empty:
    weather_df.to_csv('data/bom_weather_current.csv', index=False)
    print(f"\n=== Weather Data Summary ===")
    print(weather_df.head(10))
    print("\n=== Wind Statistics ===")
    print(weather_df[['wind_spd_kmh', 'gust_kmh']].describe())
else:
    print("No weather data downloaded")