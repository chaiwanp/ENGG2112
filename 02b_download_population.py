"""
Step 2b - Download ABS 2021 Census population data and bake it into the
           voxel grid as a ground-risk density layer.

Prerequisites: run 02_build_voxel_grid.py first.

What this does
--------------
1. Fetches SA2 boundaries from the ABS ASGS ArcGIS REST service.
2. Fetches 2021 Estimated Resident Population (ERP) by SA2 from the
   ABS Regional Statistics SDMX API.
3. Computes population density (people/km^2) for each SA2 in the study area.
4. Rasterizes the density field into a 2-D array (nx x ny) aligned with
   the voxel grid, using a vectorized geopandas spatial join.
5. Saves the density layer back into the voxel grid pickle so that
   AORVAEnv._compute_step_reward() can use real ABS data.

Outputs:
    data/population_density_sa2.gpkg        - SA2 GeoDataFrame with densities
    data/voxel_grid_westmead_liverpool.pkl  - updated (density_map added)
    outputs/population_risk_map.png         - visualisation

Risk weight function: W(rho) = tanh(rho / 5000)
See download_abs_population.py for full justification.

Usage:
    python scripts/02b_download_population.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from download_abs_population import (
    download_population_density,
    rasterize_population,
    population_risk_weight,
)
from voxel_grid_builder import VoxelGrid3D


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs('data',    exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    # --- 1. Download and save density GeoDataFrame ---
    density_gdf = download_population_density()
    density_gdf.to_file('data/population_density_sa2.gpkg', driver='GPKG')
    print("Saved data/population_density_sa2.gpkg")

    # --- 2. Rasterize into voxel grid ---
    voxel_grid  = VoxelGrid3D.load('data/voxel_grid_westmead_liverpool.pkl')
    density_map = rasterize_population(density_gdf, voxel_grid)

    voxel_grid.add_population_density(density_map)
    voxel_grid.save('data/voxel_grid_westmead_liverpool.pkl')
    print("Voxel grid updated with population density layer.")

    # --- 3. Visualise ---
    risk_map = population_risk_weight(density_map)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    im0 = axes[0].imshow(density_map.T, origin='lower', cmap='YlOrRd',
                          extent=[0, voxel_grid.nx, 0, voxel_grid.ny])
    axes[0].set_title('Population Density (people/km^2)')
    axes[0].set_xlabel('Grid X'); axes[0].set_ylabel('Grid Y')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(risk_map.T, origin='lower', cmap='RdYlGn_r',
                          vmin=0, vmax=1,
                          extent=[0, voxel_grid.nx, 0, voxel_grid.ny])
    axes[1].set_title('Ground Risk Weight  W = tanh(rho / 5000)')
    axes[1].set_xlabel('Grid X'); axes[1].set_ylabel('Grid Y')
    plt.colorbar(im1, ax=axes[1], label='W in [0, 1]')

    wx, wy, _ = voxel_grid.latlon_to_grid(-33.8078, 150.9875, 0)
    lx, ly, _ = voxel_grid.latlon_to_grid(-33.9173, 150.9233, 0)
    for ax in axes:
        ax.plot(wx, wy, 'b*', markersize=18, label='Westmead')
        ax.plot(lx, ly, 'r*', markersize=18, label='Liverpool')
        ax.legend(loc='upper right')

    plt.suptitle('ABS 2021 Census - Ground Risk Layer', fontsize=14)
    plt.tight_layout()
    plt.savefig('outputs/population_risk_map.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved outputs/population_risk_map.png")

    # --- 4. Summary ---
    print("\n=== Top 10 SA2 regions by density ===")
    print(density_gdf[['SA2_NAME_2021', 'population',
                        'density_ppl_km2', 'risk_weight']]
          .sort_values('density_ppl_km2', ascending=False)
          .head(10)
          .to_string(index=False))

    print("\nStep 2b complete.")
