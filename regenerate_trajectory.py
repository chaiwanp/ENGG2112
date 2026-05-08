"""
regenerate_trajectory.py

Regenerate the reference trajectory at the corrected cruise speed of
15 m/s (down from 20 m/s in v1). This produces a more forgiving target
arrival time of ~960 s end-to-end, which is consistent with realistic
medical delivery drone cruise speeds and gives the RL agent room to
deviate without immediately accumulating large time-deviation penalties.

Run this BEFORE retraining. The new pickle overwrites
data/reference_trajectory.pkl which the AORVAEnv loads at construction.
"""

from voxel_grid_builder import VoxelGrid3D
from trajectory_planner import (
    AStarPathfinder, compute_checkpoints, save_trajectory, visualize_trajectory
)


CRUISE_SPEED_MS = 15.0   # was 20.0 -- corrected for v2

if __name__ == "__main__":
    voxel_grid = VoxelGrid3D.load('data/voxel_grid_westmead_liverpool.pkl')

    westmead = (-33.8078, 150.9875, 100.0)
    liverpool = (-33.9173, 150.9233, 100.0)

    pathfinder = AStarPathfinder(voxel_grid, min_alt_m=50.0, max_alt_m=150.0)
    path = pathfinder.find_path(westmead, liverpool)
    if path is None:
        raise RuntimeError("No feasible path found")

    checkpoints = compute_checkpoints(
        path, voxel_grid, n_checkpoints=15, cruise_speed_ms=CRUISE_SPEED_MS
    )

    print(f"\nNew end-to-end target time: {checkpoints[-1].target_time_s:.0f} s "
          f"({checkpoints[-1].target_time_s/60:.1f} min)")

    save_trajectory(checkpoints, path, CRUISE_SPEED_MS,
                    'data/reference_trajectory.pkl')
    visualize_trajectory(voxel_grid, path, checkpoints,
                         save_path='outputs/reference_trajectory_v2.png')

    print("\nDone. Reference trajectory regenerated with 15 m/s cruise speed.")
