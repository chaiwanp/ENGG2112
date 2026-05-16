"""
Step 3 — Compute the A* reference trajectory with 15 checkpoints.

Prerequisite: run 02_build_voxel_grid.py first.

Outputs:
    data/reference_trajectory.pkl
    outputs/reference_trajectory.png

Usage:
    python scripts/03_plan_trajectory.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voxel_grid_builder import VoxelGrid3D
from trajectory_planner import (
    AStarPathfinder, compute_checkpoints,
    save_trajectory, visualize_trajectory,
)

WESTMEAD  = (-33.8078, 150.9875, 100.0)   # (lat, lon, alt_m)
LIVERPOOL = (-33.9173, 150.9233, 100.0)
CRUISE_SPEED_MS = 20.0
N_CHECKPOINTS   = 15

if __name__ == "__main__":
    os.makedirs('data',    exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    voxel_grid = VoxelGrid3D.load('data/voxel_grid_westmead_liverpool.pkl')

    pathfinder = AStarPathfinder(voxel_grid, min_alt_m=50.0, max_alt_m=150.0)
    path = pathfinder.find_path(WESTMEAD, LIVERPOOL)
    if path is None:
        raise RuntimeError("A*: no feasible path found between hospitals.")

    checkpoints = compute_checkpoints(
        path, voxel_grid,
        n_checkpoints=N_CHECKPOINTS,
        cruise_speed_ms=CRUISE_SPEED_MS,
    )

    print("\n=== Reference checkpoints ===")
    print(f"{'k':>2}  {'lat':>9}  {'lon':>9}  {'alt':>6}  "
          f"{'dist (m)':>9}  {'t (s)':>7}")
    for c in checkpoints:
        lat, lon, alt = c.latlon_alt
        print(f"{c.index:2d}  {lat:9.4f}  {lon:9.4f}  {alt:6.0f}  "
              f"{c.cumulative_distance_m:9.1f}  {c.target_time_s:7.1f}")

    save_trajectory(checkpoints, path, CRUISE_SPEED_MS,
                    'data/reference_trajectory.pkl')
    visualize_trajectory(voxel_grid, path, checkpoints,
                         save_path='outputs/reference_trajectory.png')

    print("\nStep 3 complete.")

