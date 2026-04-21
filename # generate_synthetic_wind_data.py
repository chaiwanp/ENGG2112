# generate_synthetic_wind_data.py
"""
Since BoM historical API access requires registration,
we'll create realistic synthetic wind data based on Sydney patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_realistic_wind_profile(hours=720):  # 30 days
    """
    Generate realistic wind data for Sydney based on typical patterns:
    - Mean wind speed: 15-20 km/h
    - Peak gusts: 30-50 km/h
    - Prevailing winds: NE to SE
    - Diurnal variation (stronger afternoon)
    - Weather systems (occasional high wind periods)
    """
    
    timestamps = [datetime.now() - timedelta(hours=i) for i in range(hours)]
    timestamps.reverse()
    
    data = []
    
    for i, ts in enumerate(timestamps):
        hour = ts.hour
        
        # Base wind speed with diurnal cycle
        base_speed = 15 + 5 * np.sin((hour - 6) * np.pi / 12)
        
        # Add weather system variation (3-day cycles)
        system_variation = 8 * np.sin(i * 2 * np.pi / 72)
        
        # Random turbulence
        turbulence = np.random.normal(0, 3)
        
        wind_speed = max(0, base_speed + system_variation + turbulence)
        
        # Gusts (typically 1.3-1.8x mean)
        gust_factor = np.random.uniform(1.3, 1.8)
        gust_speed = wind_speed * gust_factor
        
        # Wind direction (prevailing NE-SE, 45-135 degrees)
        base_direction = 90  # East
        direction_variation = np.random.normal(0, 30)
        wind_direction = (base_direction + direction_variation) % 360
        
        # Temperature (Sydney typical)
        temp_base = 20
        temp_daily = 8 * np.sin((hour - 14) * np.pi / 12)
        temperature = temp_base + temp_daily + np.random.normal(0, 2)
        
        data.append({
            'timestamp': ts,
            'wind_speed_kmh': round(wind_speed, 1),
            'wind_direction_deg': round(wind_direction, 0),
            'gust_speed_kmh': round(gust_speed, 1),
            'temperature_c': round(temperature, 1),
            'hour': hour
        })
    
    df = pd.DataFrame(data)
    return df

# Generate data
wind_df = generate_realistic_wind_profile(hours=720)  # 30 days
wind_df.to_csv('data/wind_historical_synthetic.csv', index=False)

print("=== Generated Wind Data Statistics ===")
print(wind_df[['wind_speed_kmh', 'gust_speed_kmh', 'wind_direction_deg']].describe())

# Visualize
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(15, 10))

# Wind speed over time
axes[0].plot(wind_df['timestamp'], wind_df['wind_speed_kmh'], label='Wind Speed', alpha=0.7)
axes[0].plot(wind_df['timestamp'], wind_df['gust_speed_kmh'], label='Gust Speed', alpha=0.5)
axes[0].set_ylabel('Speed (km/h)')
axes[0].set_title('Wind Speed Profile')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Wind direction
axes[1].scatter(wind_df['timestamp'], wind_df['wind_direction_deg'], alpha=0.3, s=10)
axes[1].set_ylabel('Direction (degrees)')
axes[1].set_title('Wind Direction')
axes[1].set_ylim(0, 360)
axes[1].grid(True, alpha=0.3)

# Hourly average
hourly_avg = wind_df.groupby('hour')['wind_speed_kmh'].mean()
axes[2].bar(hourly_avg.index, hourly_avg.values)
axes[2].set_xlabel('Hour of Day')
axes[2].set_ylabel('Avg Wind Speed (km/h)')
axes[2].set_title('Average Wind Speed by Hour')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/wind_profile_analysis.png', dpi=300, bbox_inches='tight')
plt.show()