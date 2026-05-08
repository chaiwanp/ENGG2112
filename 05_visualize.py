"""
Step 5 - Visualise voxel grid slices and wind field.

Prerequisites: run steps 00, 01, 02 first.

Outputs:
    outputs/voxel_slice_100m.png
    outputs/voxel_slice_200m.png
    outputs/wind_field_100m.png   - IDW field + every observation node
    outputs/wind_field_200m.png

Usage:
    python scripts/05_visualize.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

from voxel_grid_builder import VoxelGrid3D
from wind_field_interpolator import WindField3D
from visualize_voxel_grid import visualize_voxel_slice


def _build_wind_field(voxel_grid):
    """
    Load the best available wind data and build the WindField3D.

    Returns (wind_field, label, node_df | None).
    node_df contains the raw per-node observations if spatial data was used.
    """
    spatial_csv = 'data/wind_spatial_real.csv'
    real_csv    = 'data/wind_historical_real.csv'

    if os.path.exists(spatial_csv):
        df = pd.read_csv(spatial_csv)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        nodes = df[['lat', 'lon']].drop_duplicates()
        n = len(nodes)
        wf = WindField3D(voxel_grid, wind_data_df=df)
        wf.interpolate_wind_field()
        label = f'Real spatially-varying wind - {n} IDW observation nodes (Open-Meteo ERA5)'
        print(f"Wind source: {spatial_csv}  ({n} nodes)")
        # Latest snapshot per node for visualising observed arrows
        latest_ts = df['timestamp'].max()
        node_df = (df[df['timestamp'] == latest_ts]
                   .drop_duplicates(subset=['lat', 'lon'])
                   .copy())
        return wf, label, node_df

    if os.path.exists(real_csv):
        df = pd.read_csv(real_csv)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        wf = WindField3D(voxel_grid, wind_data_df=df)
        wf.interpolate_wind_field()
        label = ('Real single-point wind (uniform - '
                 'run  python scripts/00_download_real_wind.py  for spatial field)')
        print(f"Wind source: {real_csv}  (single-point)")
        print("  TIP: the spatial field gives a much richer plot.")
        return wf, label, None

    print("No wind data found. Run scripts/00_download_real_wind.py first.")
    return WindField3D(voxel_grid), 'No wind data', None


def _wind_components(speed_kmh, direction_deg):
    """Meteorological convention: 0 deg = from North, 90 deg = from East."""
    rad = np.radians(direction_deg)
    u = -speed_kmh * np.sin(rad)   # positive = eastward
    v = -speed_kmh * np.cos(rad)   # positive = northward
    return u, v


def visualize_wind_field_rich(wind_field, voxel_grid, node_df=None,
                               altitude_m=100, label='', out_dir='outputs'):
    """
    Wind field plot with:
      - Background speed heatmap (blue gradient)
      - White streamlines showing IDW-interpolated flow direction
      - Building footprint overlay
      - Each observation node shown as a coloured dot + direction arrow
      - Hospital markers
    """
    vs  = voxel_grid.voxel_size_m
    iz  = int(np.clip(altitude_m / vs, 0, voxel_grid.nz - 1))

    U = wind_field.u_field[:, :, iz].T    # (ny, nx)
    V = wind_field.v_field[:, :, iz].T
    speed = np.sqrt(U**2 + V**2)

    bld = voxel_grid.grid[:, :, iz].T.astype(float)
    bld[bld == 0] = np.nan

    nx, ny = voxel_grid.nx, voxel_grid.ny
    x = np.arange(nx)
    y = np.arange(ny)

    fig, ax = plt.subplots(figsize=(15, 11))

    # --- Speed heatmap ---
    vmax = max(float(speed.max()), 0.5)
    im = ax.imshow(speed, origin='lower', cmap='Blues',
                   vmin=0, vmax=vmax,
                   extent=[-0.5, nx - 0.5, -0.5, ny - 0.5],
                   aspect='auto', alpha=0.85)
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Wind speed (m/s)', fontsize=9)

    # --- Building overlay ---
    ax.imshow(bld, origin='lower', cmap='Greys', vmin=0, vmax=1,
              extent=[-0.5, nx - 0.5, -0.5, ny - 0.5],
              aspect='auto', alpha=0.40)

    # --- Streamlines (IDW-interpolated field) ---
    try:
        strm = ax.streamplot(x, y, U, V,
                             color='white', linewidth=0.8,
                             density=2.0, arrowsize=1.1,
                             broken_streamlines=False)
    except Exception:
        skip = 5
        ax.quiver(x[::skip], y[::skip],
                  U[::skip, ::skip], V[::skip, ::skip],
                  color='white', alpha=0.7, scale=50)

    # --- Observation nodes ---
    if node_df is not None and not node_df.empty:
        min_lon, min_lat, _, _ = voxel_grid.bounds
        m_per_deg_lon = voxel_grid.m_per_deg_lon
        m_per_deg_lat = voxel_grid.m_per_deg_lat

        for _, row in node_df.iterrows():
            # Convert lat/lon -> grid index
            gx = (row['lon'] - min_lon) * m_per_deg_lon / vs
            gy = (row['lat'] - min_lat) * m_per_deg_lat / vs

            spd_ms = row['wind_speed_kmh'] / 3.6
            u_obs, v_obs = _wind_components(
                row['wind_speed_kmh'], row['wind_direction_deg']
            )
            # Normalise arrow length to ~3 grid cells
            mag = np.hypot(u_obs, v_obs) + 1e-9
            arrow_len = 3.0
            du = (u_obs / mag) * arrow_len / 3.6  # back to m/s scale
            dv = (v_obs / mag) * arrow_len / 3.6

            # Dot at node location
            ax.plot(gx, gy, 'o', markersize=9, color='lime',
                    markeredgecolor='black', markeredgewidth=0.8, zorder=6)
            # Direction arrow
            ax.annotate(
                '', xy=(gx + du, gy + dv), xytext=(gx, gy),
                arrowprops=dict(arrowstyle='->', color='lime',
                                lw=1.8, mutation_scale=14),
                zorder=7
            )
            # Speed label
            ax.text(gx + 0.4, gy + 0.4, f'{spd_ms:.1f}',
                    fontsize=6.5, color='yellow', fontweight='bold', zorder=8)

    # --- Hospital markers ---
    wx, wy, _ = voxel_grid.latlon_to_grid(-33.8078, 150.9875, 0)
    lx, ly, _ = voxel_grid.latlon_to_grid(-33.9173, 150.9233, 0)
    ax.plot(wx, wy, 'b*', markersize=20, label='Westmead Hospital',
            markeredgecolor='white', markeredgewidth=1.5, zorder=9)
    ax.plot(lx, ly, 'r*', markersize=20, label='Liverpool Hospital',
            markeredgecolor='white', markeredgewidth=1.5, zorder=9)
    ax.plot([wx, lx], [wy, ly], 'w--', linewidth=1.5,
            alpha=0.6, label='Corridor', zorder=5)

    # --- Legend ---
    node_patch = mpatches.Patch(color='lime', label='Observation node (speed m/s)')
    ax.legend(handles=[
        plt.Line2D([0], [0], marker='*', color='b', markersize=12,
                   label='Westmead', linestyle='None'),
        plt.Line2D([0], [0], marker='*', color='r', markersize=12,
                   label='Liverpool', linestyle='None'),
        plt.Line2D([0], [0], color='white', linewidth=1.5,
                   label='IDW streamlines'),
        node_patch,
    ], loc='upper right', framealpha=0.85, fontsize=8)

    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)
    ax.set_xlabel('Grid X  (west -> east)', fontsize=9)
    ax.set_ylabel('Grid Y  (south -> north)', fontsize=9)
    ax.set_title(f'Wind field at {altitude_m} m altitude\n{label}', fontsize=10)

    plt.tight_layout()
    out = os.path.join(out_dir, f'wind_field_{altitude_m}m.png')
    plt.savefig(out, dpi=250, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    os.makedirs('outputs', exist_ok=True)

    voxel_grid = VoxelGrid3D.load('data/voxel_grid_westmead_liverpool.pkl')

    # All valid altitude slices: voxel_size=50 m, max_height=500 m → z=0..9
    altitudes = list(range(0, voxel_grid.nz * int(voxel_grid.voxel_size_m),
                           int(voxel_grid.voxel_size_m)))

    for alt in altitudes:
        visualize_voxel_slice(voxel_grid, altitude_m=alt)

    wind_field, label, node_df = _build_wind_field(voxel_grid)

    for alt in altitudes:
        visualize_wind_field_rich(wind_field, voxel_grid,
                                  node_df=node_df,
                                  altitude_m=alt,
                                  label=label)

    print("\nStep 5 complete.")
