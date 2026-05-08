"""
visualize_voxel_grid.py

Reusable visualisation helpers for the 3-D voxel grid and wind field.

Functions
---------
  visualize_voxel_slice(voxel_grid, altitude_m, out_dir, show)
      Horizontal occupancy slice at a given altitude.
  visualize_wind_field(wind_field, altitude_m, out_dir, show)
      Quiver plot of the wind field at a given altitude.

Import safely - no code runs at import time.
Standalone demo: python visualize_voxel_grid.py
"""

import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# Hospital (lon, lat) for marker placement
_WESTMEAD_LATLON  = (-33.8078, 150.9875)
_LIVERPOOL_LATLON = (-33.9173, 150.9233)


def visualize_voxel_slice(voxel_grid, altitude_m=100,
                           out_dir='outputs', show=False):
    """
    Horizontal occupancy slice of the voxel grid at `altitude_m`.

    Occupied voxels (buildings / no-fly zones) are shown in red/orange;
    free airspace in light blue. Hospital locations are marked.

    Parameters
    ----------
    voxel_grid  : VoxelGrid3D
    altitude_m  : float  - altitude of the horizontal slice (metres)
    out_dir     : str    - directory for the saved PNG
    show        : bool   - call plt.show() after saving (default False)
    """
    os.makedirs(out_dir, exist_ok=True)

    z_idx = int(np.clip(altitude_m / voxel_grid.voxel_size_m,
                        0, voxel_grid.nz - 1))

    occupancy = voxel_grid.grid[:, :, z_idx].T.astype(float)
    meta      = voxel_grid.metadata[:, :, z_idx].T

    # Build an RGB image: free=light-blue, building=tomato, NFZ=orange
    rgb = np.zeros((*occupancy.shape, 3), dtype=np.float32)
    free = occupancy == 0
    bld  = meta == 1
    nfz  = meta == 2

    rgb[free] = [0.85, 0.92, 1.00]   # pale blue
    rgb[bld]  = [0.89, 0.29, 0.20]   # tomato
    rgb[nfz]  = [1.00, 0.60, 0.00]   # orange

    fig, ax = plt.subplots(figsize=(13, 10))
    ax.imshow(rgb, origin='lower',
              extent=[0, voxel_grid.nx, 0, voxel_grid.ny],
              aspect='auto', interpolation='nearest')

    # Hospital markers
    for (lat, lon), colour, name in [
        (_WESTMEAD_LATLON,  'blue',  'Westmead Hospital'),
        (_LIVERPOOL_LATLON, 'green', 'Liverpool Hospital'),
    ]:
        gx, gy, _ = voxel_grid.latlon_to_grid(lat, lon, 0)
        ax.plot(gx, gy, '*', color=colour, markersize=22,
                markeredgecolor='white', markeredgewidth=1.8,
                label=name, zorder=9)

    # Legend patches
    legend_handles = [
        mpatches.Patch(color=[0.85, 0.92, 1.00], label='Free airspace'),
        mpatches.Patch(color=[0.89, 0.29, 0.20], label='Building (occupied)'),
        mpatches.Patch(color=[1.00, 0.60, 0.00], label='No-fly zone'),
        plt.Line2D([0], [0], marker='*', color='blue',   markersize=12,
                   label='Westmead Hospital',  linestyle='None'),
        plt.Line2D([0], [0], marker='*', color='green',  markersize=12,
                   label='Liverpool Hospital', linestyle='None'),
    ]
    ax.legend(handles=legend_handles, loc='upper right',
              framealpha=0.9, fontsize=9)

    occ_pct = 100.0 * np.sum(occupancy) / occupancy.size
    ax.set_title(
        f'Voxel Grid - Horizontal Slice at {altitude_m:.0f} m altitude\n'
        f'{voxel_grid.nx}x{voxel_grid.ny} grid, {voxel_grid.voxel_size_m:.0f} m voxels, '
        f'{occ_pct:.1f}% occupied',
        fontsize=11,
    )
    ax.set_xlabel('Grid X  (west -> east)', fontsize=9)
    ax.set_ylabel('Grid Y  (south -> north)', fontsize=9)

    plt.tight_layout()
    out = os.path.join(out_dir, f'voxel_slice_{altitude_m:.0f}m.png')
    plt.savefig(out, dpi=250, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved {out}")


def visualize_wind_field(wind_field, altitude_m=100,
                          out_dir='outputs', show=False):
    """
    Quiver plot of u/v wind vectors at `altitude_m`.

    Arrows are colour-coded by speed magnitude.  Hospital markers
    and a grid line overlay are included.

    Parameters
    ----------
    wind_field  : WindField3D
    altitude_m  : float  - altitude slice (metres)
    out_dir     : str    - directory for the saved PNG
    show        : bool   - call plt.show() after saving
    """
    os.makedirs(out_dir, exist_ok=True)

    vg   = wind_field.voxel_grid
    z_idx = int(np.clip(altitude_m / vg.voxel_size_m, 0, vg.nz - 1))

    u_slice = wind_field.u_field[:, :, z_idx]   # (nx, ny)
    v_slice = wind_field.v_field[:, :, z_idx]
    speed   = np.sqrt(u_slice**2 + v_slice**2)

    skip = max(1, vg.nx // 30)   # ~30 arrows across
    X, Y = np.meshgrid(
        np.arange(0, vg.nx, skip),
        np.arange(0, vg.ny, skip),
    )
    U = u_slice[::skip, ::skip].T
    V = v_slice[::skip, ::skip].T
    C = speed[::skip, ::skip].T

    fig, ax = plt.subplots(figsize=(13, 10))

    # Speed background
    im = ax.imshow(speed.T, origin='lower', cmap='Blues', alpha=0.5,
                   extent=[0, vg.nx, 0, vg.ny], aspect='auto')
    plt.colorbar(im, ax=ax, label='Wind speed (m/s)', fraction=0.03, pad=0.02)

    # Wind vectors
    Q = ax.quiver(X, Y, U, V, C, cmap='coolwarm', scale=None,
                  pivot='mid', width=0.003, headwidth=4)

    # Hospital markers
    for (lat, lon), colour, name in [
        (_WESTMEAD_LATLON,  'blue',  'Westmead Hospital'),
        (_LIVERPOOL_LATLON, 'limegreen', 'Liverpool Hospital'),
    ]:
        gx, gy, _ = vg.latlon_to_grid(lat, lon, 0)
        ax.plot(gx, gy, '*', color=colour, markersize=22,
                markeredgecolor='white', markeredgewidth=1.8,
                label=name, zorder=9)

    mean_speed = float(speed.mean())
    max_speed  = float(speed.max())
    ax.set_title(
        f'Wind Field at {altitude_m:.0f} m altitude  '
        f'(mean {mean_speed:.1f} m/s, max {max_speed:.1f} m/s, '
        f'alpha={wind_field.alpha})',
        fontsize=11,
    )
    ax.set_xlabel('Grid X  (west -> east)',   fontsize=9)
    ax.set_ylabel('Grid Y  (south -> north)', fontsize=9)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.2, linestyle='--')

    plt.tight_layout()
    out = os.path.join(out_dir, f'wind_field_{altitude_m:.0f}m.png')
    plt.savefig(out, dpi=250, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved {out}")


# ======================================================================
# Standalone demo  (python visualize_voxel_grid.py)
# ======================================================================
if __name__ == "__main__":
    import pandas as pd
    from voxel_grid_builder import VoxelGrid3D
    from wind_field_interpolator import WindField3D

    os.makedirs('outputs', exist_ok=True)

    print("Loading voxel grid...")
    voxel_grid = VoxelGrid3D.load('data/voxel_grid_westmead_liverpool.pkl')

    for alt in [100, 200]:
        visualize_voxel_slice(voxel_grid, altitude_m=alt)

    # Load wind data (prefer spatial, fall back to single-point)
    for csv_path in ['data/wind_spatial_real.csv', 'data/wind_historical_real.csv']:
        if os.path.exists(csv_path):
            print(f"Loading wind data from {csv_path}")
            wind_df = pd.read_csv(csv_path)
            wind_df['timestamp'] = pd.to_datetime(wind_df['timestamp'])
            wf = WindField3D(voxel_grid, wind_data_df=wind_df)
            wf.interpolate_wind_field()
            for alt in [100, 200]:
                visualize_wind_field(wf, altitude_m=alt)
            break
    else:
        print("No wind CSV found. Run scripts/00_download_real_wind.py first.")
