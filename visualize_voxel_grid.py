"""
visualize_voxel_grid.py

Helpers for plotting 2D slices of a VoxelGrid3D.
Imported by scripts/05_visualise.py.

Functions
---------
    visualize_voxel_slice(voxel_grid, altitude_m, out_dir)
        Horizontal occupancy slice at a given altitude with hospital markers.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def visualize_voxel_slice(voxel_grid,
                           altitude_m: float = 100.0,
                           out_dir: str = "outputs") -> None:
    """
    Save a top-down occupancy slice of the voxel grid at `altitude_m`.

    Occupied voxels (buildings / no-fly zones) are shown in dark grey.
    Hospital markers and the corridor line are overlaid.

    Parameters
    ----------
    voxel_grid : VoxelGrid3D
    altitude_m : float
        Altitude of the slice in metres.
    out_dir : str
        Directory to save the PNG.
    """
    os.makedirs(out_dir, exist_ok=True)

    vs   = voxel_grid.voxel_size_m
    vs_z = getattr(voxel_grid, "voxel_size_z_m", vs)
    iz   = int(np.clip(altitude_m / vs_z, 0, voxel_grid.nz - 1))

    # 2D occupancy slice (transposed so Y is up)
    slice_2d = voxel_grid.grid[:, :, iz].T

    fig, ax = plt.subplots(figsize=(13, 10))
    ax.imshow(
        slice_2d, origin="lower", cmap="Greys",
        vmin=0, vmax=1,
        extent=[-0.5, voxel_grid.nx - 0.5, -0.5, voxel_grid.ny - 0.5],
        aspect="auto",
    )

    # Metadata overlay if available (1=building, 2=no-fly zone)
    if hasattr(voxel_grid, "metadata"):
        meta = voxel_grid.metadata[:, :, iz].T.astype(float)
        nfz  = np.where(meta == 2, 1.0, np.nan)
        ax.imshow(
            nfz, origin="lower", cmap="Reds", alpha=0.5,
            extent=[-0.5, voxel_grid.nx - 0.5, -0.5, voxel_grid.ny - 0.5],
            aspect="auto",
        )

    # Hospital markers
    wx, wy, _ = voxel_grid.latlon_to_grid(-33.8078, 150.9875, 0)
    lx, ly, _ = voxel_grid.latlon_to_grid(-33.9173, 150.9233, 0)
    ax.plot(wx, wy, "b*", markersize=22, markeredgecolor="white",
            markeredgewidth=1.5, label="Westmead Hospital", zorder=6)
    ax.plot(lx, ly, "r*", markersize=22, markeredgecolor="white",
            markeredgewidth=1.5, label="Liverpool Hospital", zorder=6)
    ax.plot([wx, lx], [wy, ly], "w--", linewidth=1.5,
            alpha=0.7, label="Corridor", zorder=5)

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="black",  label="Building"),
        Patch(facecolor="red",    alpha=0.5, label="No-fly zone (NFZ)"),
        plt.Line2D([0], [0], marker="*", color="b", markersize=12,
                   label="Westmead", linestyle="None"),
        plt.Line2D([0], [0], marker="*", color="r", markersize=12,
                   label="Liverpool", linestyle="None"),
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              fontsize=9, framealpha=0.9)

    occupied   = int(np.sum(slice_2d > 0))
    total_2d   = voxel_grid.nx * voxel_grid.ny
    occ_pct    = 100 * occupied / total_2d

    ax.set_title(
        f"Voxel grid occupancy at {altitude_m:.0f} m altitude "
        f"(z-index {iz})  —  {occ_pct:.1f}% occupied",
        fontsize=11,
    )
    ax.set_xlabel("Grid X  (west → east)",   fontsize=9)
    ax.set_ylabel("Grid Y  (south → north)", fontsize=9)
    ax.set_xlim(-0.5, voxel_grid.nx - 0.5)
    ax.set_ylim(-0.5, voxel_grid.ny - 0.5)

    plt.tight_layout()
    out = os.path.join(out_dir, f"voxel_slice_{altitude_m:.0f}m.png")
    plt.savefig(out, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")
