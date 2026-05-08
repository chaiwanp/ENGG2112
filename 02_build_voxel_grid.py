"""
Step 2 — Build the 3-D voxel occupancy grid from the downloaded buildings.

Prerequisite: run 01_download_buildings.py first.

Output: data/voxel_grid_westmead_liverpool.pkl

Usage:
    python scripts/02_build_voxel_grid.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import geopandas as gpd
from voxel_grid_builder import VoxelGrid3D

# Bounding box: Westmead to Liverpool + 5 km buffer
BOUNDS = (150.8233, -34.0173, 151.0875, -33.7078)  # min_lon, min_lat, max_lon, max_lat

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)

    grid = VoxelGrid3D(bounds=BOUNDS, voxel_size_m=50, max_height_m=500)

    for candidate in [
        'data/buildings_westmead_liverpool.gpkg',
        'data/buildings_westmead_liverpool.geojson',
        'data/buildings_westmead_liverpool.pkl',
    ]:
        if os.path.exists(candidate):
            print(f"Loading buildings from {candidate}")
            if candidate.endswith('.pkl'):
                with open(candidate, 'rb') as f:
                    buildings = pickle.load(f)
            else:
                buildings = gpd.read_file(candidate)
            break
    else:
        raise FileNotFoundError(
            "No buildings file found. Run 01_download_buildings.py first."
        )

    grid.add_buildings(buildings)

    # Add Bankstown Airport CTR (CASA CASR Part 101 compliance)
    grid.add_bankstown_nfz()

    grid.save('data/voxel_grid_westmead_liverpool.pkl')
    print("\nStep 2 complete.")
