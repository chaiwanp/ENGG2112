# voxel_grid_builder.py
import os
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, box
import pickle

_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_DIR, 'data')
_OUTPUT_DIR = os.path.join(_DIR, 'outputs')

class VoxelGrid3D:
    def __init__(self, bounds, voxel_size_m=50, max_height_m=500):
        """
        Create 3D voxel grid
        
        Parameters:
        - bounds: (min_lon, min_lat, max_lon, max_lat)
        - voxel_size_m: size of each voxel in meters
        - max_height_m: maximum altitude
        """
        self.voxel_size_m = voxel_size_m
        self.max_height_m = max_height_m
        self.bounds = bounds
        
        # Convert lat/lon bounds to approximate meters
        min_lon, min_lat, max_lon, max_lat = bounds
        
        # Approximate meters per degree at Sydney latitude
        lat_center = (min_lat + max_lat) / 2
        m_per_deg_lat = 111320
        m_per_deg_lon = 111320 * np.cos(np.radians(lat_center))
        
        # Grid dimensions
        self.width_m = (max_lon - min_lon) * m_per_deg_lon
        self.length_m = (max_lat - min_lat) * m_per_deg_lat
        
        self.nx = int(np.ceil(self.width_m / voxel_size_m))
        self.ny = int(np.ceil(self.length_m / voxel_size_m))
        self.nz = int(np.ceil(max_height_m / voxel_size_m))
        
        print(f"Grid dimensions: {self.nx} x {self.ny} x {self.nz}")
        print(f"Total voxels: {self.nx * self.ny * self.nz:,}")
        
        # Initialize grid (0 = free space, 1 = occupied)
        self.grid = np.zeros((self.nx, self.ny, self.nz), dtype=np.uint8)
        
        # Metadata grid for additional info
        self.metadata = np.zeros((self.nx, self.ny, self.nz), dtype=np.uint8)
        # 0=free, 1=building, 2=no-fly zone, 3=high-risk ground
        
        self.m_per_deg_lat = m_per_deg_lat
        self.m_per_deg_lon = m_per_deg_lon
        
    def latlon_to_grid(self, lat, lon, alt_m=0):
        """Convert lat/lon/altitude to grid indices"""
        min_lon, min_lat, max_lon, max_lat = self.bounds
        
        x = int((lon - min_lon) * self.m_per_deg_lon / self.voxel_size_m)
        y = int((lat - min_lat) * self.m_per_deg_lat / self.voxel_size_m)
        z = int(alt_m / self.voxel_size_m)
        
        # Clamp to grid bounds
        x = np.clip(x, 0, self.nx - 1)
        y = np.clip(y, 0, self.ny - 1)
        z = np.clip(z, 0, self.nz - 1)
        
        return x, y, z
    
    def grid_to_latlon(self, x, y, z):
        """Convert grid indices to lat/lon/altitude"""
        min_lon, min_lat, max_lon, max_lat = self.bounds
        
        lon = min_lon + (x * self.voxel_size_m) / self.m_per_deg_lon
        lat = min_lat + (y * self.voxel_size_m) / self.m_per_deg_lat
        alt_m = z * self.voxel_size_m
        
        return lat, lon, alt_m
    
    def add_buildings(self, buildings_gdf):
        """Add buildings to voxel grid"""
        print(f"Adding {len(buildings_gdf)} buildings to grid...")
        
        for idx, building in buildings_gdf.iterrows():
            geom = building.geometry
            height = building['height_m']
            
            # Get building bounds
            minx, miny, maxx, maxy = geom.bounds
            
            # Convert to grid coordinates
            x_start, y_start, _ = self.latlon_to_grid(miny, minx, 0)
            x_end, y_end, _ = self.latlon_to_grid(maxy, maxx, 0)
            z_max = int(height / self.voxel_size_m)
            
            # Fill voxels
            for x in range(x_start, min(x_end + 1, self.nx)):
                for y in range(y_start, min(y_end + 1, self.ny)):
                    for z in range(0, min(z_max, self.nz)):
                        self.grid[x, y, z] = 1
                        self.metadata[x, y, z] = 1  # Building
            
            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx + 1} buildings...")
        
        occupied = np.sum(self.grid)
        total = self.grid.size
        print(f"Occupied voxels: {occupied:,} ({100*occupied/total:.2f}%)")
    
    def add_no_fly_zones(self, zones_gdf, min_alt_m=0, max_alt_m=500):
        """Add no-fly zones (CASA restricted areas)"""
        print(f"Adding {len(zones_gdf)} no-fly zones...")
        
        for idx, zone in zones_gdf.iterrows():
            geom = zone.geometry
            minx, miny, maxx, maxy = geom.bounds
            
            x_start, y_start, _ = self.latlon_to_grid(miny, minx, 0)
            x_end, y_end, _ = self.latlon_to_grid(maxy, maxx, 0)
            z_start = int(min_alt_m / self.voxel_size_m)
            z_end = int(max_alt_m / self.voxel_size_m)
            
            for x in range(x_start, min(x_end + 1, self.nx)):
                for y in range(y_start, min(y_end + 1, self.ny)):
                    for z in range(z_start, min(z_end, self.nz)):
                        self.grid[x, y, z] = 1
                        self.metadata[x, y, z] = 2  # No-fly zone
    
    def save(self, filepath):
        """Save voxel grid to file"""
        data = {
            'grid': self.grid,
            'metadata': self.metadata,
            'bounds': self.bounds,
            'voxel_size_m': self.voxel_size_m,
            'max_height_m': self.max_height_m,
            'dimensions': (self.nx, self.ny, self.nz),
            'm_per_deg_lat': self.m_per_deg_lat,
            'm_per_deg_lon': self.m_per_deg_lon
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved voxel grid to {filepath}")
        print(f"File size: {os.path.getsize(filepath) / 1024 / 1024:.2f} MB")
    
    @classmethod
    def load(cls, filepath):
        """Load voxel grid from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Reconstruct object
        grid_obj = cls.__new__(cls)
        grid_obj.grid = data['grid']
        grid_obj.metadata = data['metadata']
        grid_obj.bounds = data['bounds']
        grid_obj.voxel_size_m = data['voxel_size_m']
        grid_obj.max_height_m = data['max_height_m']
        grid_obj.nx, grid_obj.ny, grid_obj.nz = data['dimensions']
        grid_obj.m_per_deg_lat = data['m_per_deg_lat']
        grid_obj.m_per_deg_lon = data['m_per_deg_lon']
        grid_obj.width_m = grid_obj.nx * data['voxel_size_m']
        grid_obj.length_m = grid_obj.ny * data['voxel_size_m']

        return grid_obj

# Build the voxel grid
if __name__ == '__main__':
    os.makedirs(_DATA_DIR, exist_ok=True)
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    # Define bounds: Westmead to Liverpool with 5km buffer
    bounds = (150.8233, -34.0173, 151.0875, -33.7078)

    # Create voxel grid (50m resolution, up to 500m altitude)
    voxel_grid = VoxelGrid3D(
        bounds=bounds,
        voxel_size_m=50,
        max_height_m=500
    )

    # Load and add buildings
    buildings = gpd.read_file(os.path.join(_DATA_DIR, 'buildings_westmead_liverpool.gpkg'))
    voxel_grid.add_buildings(buildings)

    # Save
    voxel_grid.save(os.path.join(_DATA_DIR, 'voxel_grid_westmead_liverpool.pkl'))