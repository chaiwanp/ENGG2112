# wind_field_interpolator.py
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, griddata

class WindField3D:
    def __init__(self, voxel_grid, wind_data_df):
        """
        Create 3D wind field from weather station data
        
        Parameters:
        - voxel_grid: VoxelGrid3D object
        - wind_data_df: DataFrame with columns: 
          ['timestamp', 'wind_speed_kmh', 'wind_direction_deg', 'altitude_m']
        """
        self.voxel_grid = voxel_grid
        self.wind_data = wind_data_df
        
        # Wind components (u, v, w)
        self.u_field = np.zeros((voxel_grid.nx, voxel_grid.ny, voxel_grid.nz))
        self.v_field = np.zeros((voxel_grid.nx, voxel_grid.ny, voxel_grid.nz))
        self.w_field = np.zeros((voxel_grid.nx, voxel_grid.ny, voxel_grid.nz))
        
    def convert_wind_to_components(self, speed_kmh, direction_deg):
        """
        Convert wind speed and direction to u, v components
        
        Direction: meteorological convention (direction FROM which wind blows)
        0° = North, 90° = East, 180° = South, 270° = West
        """
        # Convert to m/s
        speed_ms = speed_kmh / 3.6
        
        # Convert to radians (meteorological to mathematical)
        # Meteorological: 0° = from North, clockwise
        # Mathematical: 0° = East (positive x), counterclockwise
        theta_rad = np.radians(270 - direction_deg)
        
        # u = eastward component, v = northward component
        u = speed_ms * np.cos(theta_rad)
        v = speed_ms * np.sin(theta_rad)
        
        return u, v
    
    def interpolate_wind_field(self, timestamp=None):
        """
        Interpolate wind field across entire grid
        
        Uses power law for vertical wind profile:
        v(z) = v_ref * (z / z_ref)^alpha
        where alpha ≈ 0.15 for urban areas
        """
        
        if timestamp is None:
            # Use most recent data
            wind_sample = self.wind_data.iloc[-1]
        else:
            # Find closest timestamp
            idx = (self.wind_data['timestamp'] - timestamp).abs().idxmin()
            wind_sample = self.wind_data.loc[idx]
        
        speed_kmh = wind_sample['wind_speed_kmh']
        direction_deg = wind_sample['wind_direction_deg']
        
        # Convert to components at reference height (10m)
        u_ref, v_ref = self.convert_wind_to_components(speed_kmh, direction_deg)
        z_ref = 10.0  # Reference height in meters
        alpha = 0.15  # Urban terrain roughness
        
        # Fill 3D grid
        for iz in range(self.voxel_grid.nz):
            z = (iz + 0.5) * self.voxel_grid.voxel_size_m
            
            # Power law scaling
            if z > 0:
                scale_factor = (z / z_ref) ** alpha
            else:
                scale_factor = 0
            
            u_scaled = u_ref * scale_factor
            v_scaled = v_ref * scale_factor
            
            # Apply to all horizontal positions
            self.u_field[:, :, iz] = u_scaled
            self.v_field[:, :, iz] = v_scaled
            self.w_field[:, :, iz] = 0  # Assume no vertical wind
        
        print(f"Wind field interpolated: {speed_kmh:.1f} km/h from {direction_deg:.0f}°")
        print(f"At 100m altitude: {np.mean(self.u_field[:,:,2]):.2f} m/s (u), {np.mean(self.v_field[:,:,2]):.2f} m/s (v)")
    
    def get_wind_at_position(self, x, y, z):
        """Get wind components at specific grid position"""
        x = int(np.clip(x, 0, self.voxel_grid.nx - 1))
        y = int(np.clip(y, 0, self.voxel_grid.ny - 1))
        z = int(np.clip(z, 0, self.voxel_grid.nz - 1))
        
        return self.u_field[x, y, z], self.v_field[x, y, z], self.w_field[x, y, z]
    
    def get_wind_magnitude(self, x, y, z):
        """Get wind magnitude at position"""
        u, v, w = self.get_wind_at_position(x, y, z)
        return np.sqrt(u**2 + v**2 + w**2)
    
    def save(self, filepath):
        """Save wind field"""
        data = {
            'u_field': self.u_field,
            'v_field': self.v_field,
            'w_field': self.w_field
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved wind field to {filepath}")

# Create wind field
wind_df = pd.read_csv('data/wind_historical_synthetic.csv')
wind_df['timestamp'] = pd.to_datetime(wind_df['timestamp'])

voxel_grid = VoxelGrid3D.load('data/voxel_grid_westmead_liverpool.pkl')

wind_field = WindField3D(voxel_grid, wind_df)
wind_field.interpolate_wind_field()  # Use latest data
wind_field.save('data/wind_field_current.pkl')