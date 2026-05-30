"""
trajectory_planner.py

Reference trajectory computation for the AORVA drone.

This module implements the Week-8 deliverable from the project proposal:
    "the shortest feasible path is calculated, accounting for all avoided
     areas, by using A* path-finding on the Voxel grid. This is divided
     into 15 equally spaced checkpoints, and the target arrival time is
     calculated from the segments distance and the drone's cruise speed
     under zero-wind-conditions."

Pipeline
--------
1. AStarPathfinder.find_path(start, goal)
       -> list of (x, y, z) voxel tuples, shortest feasible path that
          avoids buildings + no-fly zones and respects an altitude band.

2. compute_checkpoints(path, voxel_grid, n=15, cruise_speed_ms=20)
       -> list of Checkpoint objects with (voxel, lat/lon/alt,
          cumulative distance, target arrival time).

3. visualize_trajectory(...) draws top-down and side-profile views.

The checkpoints are the reference timeline used by the RL reward function:
for each checkpoint k,
        T_k = d_k / v_cruise
and the reward penalises deviation between the agent's actual arrival
time at checkpoint k and T_k.
"""

from __future__ import annotations

import heapq
import pickle
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# 26-connected 3D neighbourhood
_NEIGHBOR_OFFSETS: List[Tuple[int, int, int]] = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
]

# ======================================================================
# A* pathfinder
# ======================================================================
class AStarPathfinder:
    """
    3D A* on a VoxelGrid3D.

    A voxel (x, y, z) is traversable iff
        voxel_grid.grid[x, y, z] == 0   (not a building, not a no-fly zone)
        AND min_alt_idx <= z <= max_alt_idx
    """

    def __init__(self, voxel_grid, min_alt_m: float = 25.0,
                 max_alt_m: float = 150.0):
        self.voxel_grid = voxel_grid
        self.grid = voxel_grid.grid
        self.vs = voxel_grid.voxel_size_m
        self.vsz = voxel_grid.voxel_size_z_m
        self.nx, self.ny, self.nz = voxel_grid.nx, voxel_grid.ny, voxel_grid.nz

        # Use voxel_size_z_m (not voxel_size_m) to convert altitude -> z index
        self.min_z = max(0, int(min_alt_m / self.vsz))
        self.max_z = min(self.nz - 1, int(max_alt_m / self.vsz))

        if self.min_z > self.max_z:
            raise ValueError(
                f"min_alt {min_alt_m} m exceeds max_alt {max_alt_m} m for this grid"
            )

        # Precompute neighbour costs in real metres (XY uses vs, Z uses vsz)
        self._neighbor_costs: List[float] = [
            float(np.sqrt((dx * self.vs) ** 2 + (dy * self.vs) ** 2 + (dz * self.vsz) ** 2))
            for (dx, dy, dz) in _NEIGHBOR_OFFSETS
        ]

    # --- traversability helpers -------------------------------------------
    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.nx and 0 <= y < self.ny

    def _traversable(self, x: int, y: int, z: int) -> bool:
        if not self._in_bounds(x, y):
            return False
        if z < self.min_z or z > self.max_z:
            return False
        return self.grid[x, y, z] == 0

    def _heuristic(self, a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
        dx = (a[0] - b[0]) * self.vs
        dy = (a[1] - b[1]) * self.vs
        dz = (a[2] - b[2]) * self.vsz
        return float(np.sqrt(dx * dx + dy * dy + dz * dz))

    def _snap_to_free(self, voxel: Tuple[int, int, int]
                      ) -> Optional[Tuple[int, int, int]]:
        """Find nearest traversable voxel if `voxel` is blocked."""
        x, y, z = voxel
        z = max(self.min_z, min(self.max_z, z))

        if self._traversable(x, y, z):
            return (x, y, z)

        # Spiral perimeter search in the x-y plane at given z
        for radius in range(1, 40):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) != radius and abs(dy) != radius:
                        continue  # skip interior (already checked in prior radii)
                    cand = (x + dx, y + dy, z)
                    if self._traversable(*cand):
                        return cand

        # Fallback: try other altitudes at same x,y
        for dz in range(1, self.max_z - self.min_z + 1):
            for sign in (1, -1):
                cz = z + sign * dz
                if self.min_z <= cz <= self.max_z and self._traversable(x, y, cz):
                    return (x, y, cz)

        return None

    # --- main search ------------------------------------------------------
    def find_path(self,
                  start_latlon_alt: Tuple[float, float, float],
                  goal_latlon_alt: Tuple[float, float, float]
                  ) -> Optional[List[Tuple[int, int, int]]]:
        """
        Parameters
        ----------
        start_latlon_alt, goal_latlon_alt : (lat, lon, alt_m)

        Returns
        -------
        path : list of (x, y, z) voxel coordinates from start to goal,
               or None if no feasible path exists.
        """
        raw_start = self.voxel_grid.latlon_to_grid(*start_latlon_alt)
        raw_goal = self.voxel_grid.latlon_to_grid(*goal_latlon_alt)

        start = self._snap_to_free(raw_start)
        goal = self._snap_to_free(raw_goal)

        if start is None or goal is None:
            print("A*: start or goal has no reachable free voxel in altitude band")
            return None

        if start != raw_start:
            print(f"A*: start snapped {raw_start} -> {start}")
        if goal != raw_goal:
            print(f"A*: goal snapped  {raw_goal} -> {goal}")

        print(f"A*: searching {start} -> {goal}")
        print(f"    altitude band: z={self.min_z}..{self.max_z} "
              f"({self.min_z * self.vs:.0f}-{self.max_z * self.vs:.0f} m)")

        open_heap: list = []
        heapq.heappush(open_heap, (self._heuristic(start, goal), 0.0, start))
        came_from: dict = {}
        g_score: dict = {start: 0.0}
        closed: set = set()

        explored = 0
        while open_heap:
            _, g, current = heapq.heappop(open_heap)

            if current in closed:
                continue
            closed.add(current)
            explored += 1

            if current == goal:
                path = self._reconstruct(came_from, current)
                print(f"A*: path found, {len(path)} voxels, {explored} explored")
                return path

            cx, cy, cz = current
            for i, (dx, dy, dz) in enumerate(_NEIGHBOR_OFFSETS):
                nb = (cx + dx, cy + dy, cz + dz)
                if nb in closed:
                    continue
                if not self._traversable(*nb):
                    continue

                tentative_g = g + self._neighbor_costs[i]
                if tentative_g < g_score.get(nb, float('inf')):
                    g_score[nb] = tentative_g
                    f = tentative_g + self._heuristic(nb, goal)
                    came_from[nb] = current
                    heapq.heappush(open_heap, (f, tentative_g, nb))

        print(f"A*: no path after exploring {explored} voxels")
        return None

    @staticmethod
    def _reconstruct(came_from: dict,
                     current: Tuple[int, int, int]
                     ) -> List[Tuple[int, int, int]]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


# ======================================================================
# Checkpoint generation
# ======================================================================
@dataclass
class Checkpoint:
    """A single reference waypoint with a target arrival time."""
    index: int
    voxel: Tuple[int, int, int]
    latlon_alt: Tuple[float, float, float]
    cumulative_distance_m: float
    target_time_s: float


def compute_checkpoints(path_voxels: List[Tuple[int, int, int]],
                        voxel_grid,
                        n_checkpoints: int = 15,
                        cruise_speed_ms: float = 20.0
                        ) -> List[Checkpoint]:
    """
    Divide an A* path into equally-spaced checkpoints along its arc length
    and assign a target arrival time at each.

    Parameters
    ----------
    path_voxels : output of AStarPathfinder.find_path()
    voxel_grid  : VoxelGrid3D
    n_checkpoints : number of checkpoints, inclusive of start and goal
    cruise_speed_ms : zero-wind cruise speed in m/s

    Returns
    -------
    list of Checkpoint, length n_checkpoints.
    """
    if len(path_voxels) < 2:
        raise ValueError("Path must have at least 2 voxels")
    if n_checkpoints < 2:
        raise ValueError("Need at least 2 checkpoints (start and goal)")

    vs = voxel_grid.voxel_size_m
    vsz = voxel_grid.voxel_size_z_m
    pts = np.asarray(path_voxels, dtype=np.float64)

    # Cumulative arc length in metres — XY and Z have different voxel scales
    diffs = np.diff(pts, axis=0)
    seg_lengths = np.sqrt(
        (diffs[:, 0] * vs) ** 2 + (diffs[:, 1] * vs) ** 2 + (diffs[:, 2] * vsz) ** 2
    )
    cumulative = np.concatenate(([0.0], np.cumsum(seg_lengths)))
    total_distance = float(cumulative[-1])
    total_time = total_distance / cruise_speed_ms

    print(f"Path length: {total_distance:.1f} m")
    print(f"Flight time at {cruise_speed_ms} m/s (zero wind): {total_time:.1f} s")

    # Target cumulative distances for each checkpoint
    targets_m = np.linspace(0.0, total_distance, n_checkpoints)

    checkpoints: List[Checkpoint] = []
    for k, target_m in enumerate(targets_m):
        idx = int(np.searchsorted(cumulative, target_m, side='left'))
        idx = min(idx, len(path_voxels) - 1)
        vx = path_voxels[idx]
        lat, lon, alt_m = voxel_grid.grid_to_latlon(*vx)
        checkpoints.append(Checkpoint(
            index=k,
            voxel=tuple(vx),
            latlon_alt=(lat, lon, alt_m),
            cumulative_distance_m=float(target_m),
            target_time_s=float(target_m / cruise_speed_ms),
        ))

    return checkpoints


# ======================================================================
# Visualisation
# ======================================================================
def visualize_trajectory(voxel_grid,
                         path_voxels: List[Tuple[int, int, int]],
                         checkpoints: List[Checkpoint],
                         save_path: Optional[str] = None) -> None:
    """Top-down + side-profile views of the path with checkpoints."""
    import matplotlib.pyplot as plt

    path = np.asarray(path_voxels)
    vs = voxel_grid.voxel_size_m

    fig, (ax_top, ax_side) = plt.subplots(1, 2, figsize=(18, 8))

    # --- Top-down: project obstacles through the path altitude band ---
    max_z_idx = int(min(path[:, 2].max() + 1, voxel_grid.nz))
    occ_projection = np.max(voxel_grid.grid[:, :, :max_z_idx], axis=2).T

    ax_top.imshow(occ_projection, origin='lower', cmap='Greys', alpha=0.5,
                  extent=[0, voxel_grid.nx, 0, voxel_grid.ny])

    sc = ax_top.scatter(path[:, 0], path[:, 1],
                        c=path[:, 2] * vs, s=2, cmap='viridis')
    plt.colorbar(sc, ax=ax_top, label='Altitude (m)')

    cp = np.array([c.voxel for c in checkpoints])
    ax_top.plot(cp[:, 0], cp[:, 1], 'o',
                color='red', markersize=8, markeredgecolor='white',
                linewidth=0, label='Checkpoints')
    ax_top.plot(path[0, 0], path[0, 1], '*', color='blue',
                markersize=22, markeredgecolor='white', label='Westmead')
    ax_top.plot(path[-1, 0], path[-1, 1], '*', color='lime',
                markersize=22, markeredgecolor='black', label='Liverpool')

    ax_top.set_xlabel('Grid X')
    ax_top.set_ylabel('Grid Y')
    ax_top.set_title('Top-down: A* path with checkpoints')
    ax_top.legend(loc='upper right')

    # --- Side profile: altitude vs cumulative distance ---
    vsz = voxel_grid.voxel_size_z_m
    pts = path.astype(np.float64)
    diffs = np.diff(pts, axis=0)
    seg_lengths = np.sqrt(
        (diffs[:, 0] * vs) ** 2 + (diffs[:, 1] * vs) ** 2 + (diffs[:, 2] * vsz) ** 2
    )
    cumulative = np.concatenate(([0.0], np.cumsum(seg_lengths)))
    alts = path[:, 2] * vsz  # z_index * voxel_size_z_m = real altitude in metres

    ax_side.plot(cumulative, alts, '-', color='steelblue', linewidth=1.2,
                 alpha=0.8, label='Path altitude')

    cp_d = [c.cumulative_distance_m for c in checkpoints]
    cp_a = [c.latlon_alt[2] for c in checkpoints]
    ax_side.plot(cp_d, cp_a, 'o', color='red', markersize=8,
                 markeredgecolor='white', label='Checkpoints')

    for c in checkpoints:
        ax_side.annotate(f'{c.index}\nt={c.target_time_s:.0f}s',
                         (c.cumulative_distance_m, c.latlon_alt[2]),
                         textcoords='offset points', xytext=(0, 10),
                         fontsize=7, ha='center')

    ax_side.set_xlabel('Cumulative distance (m)')
    ax_side.set_ylabel('Altitude (m)')
    ax_side.set_title('Side profile: altitude vs distance, with target times')
    ax_side.grid(True, alpha=0.3)
    ax_side.legend(loc='upper right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved {save_path}")
    plt.close(fig)


# ======================================================================
# Persistence
# ======================================================================
def save_trajectory(checkpoints: List[Checkpoint],
                    path_voxels: List[Tuple[int, int, int]],
                    cruise_speed_ms: float,
                    filepath: str) -> None:
    data = {
        'path_voxels': path_voxels,
        'cruise_speed_ms': cruise_speed_ms,
        'checkpoints': [
            {
                'index': c.index,
                'voxel': c.voxel,
                'latlon_alt': c.latlon_alt,
                'cumulative_distance_m': c.cumulative_distance_m,
                'target_time_s': c.target_time_s,
            }
            for c in checkpoints
        ],
    }
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved reference trajectory to {filepath}")


def load_trajectory(filepath: str):
    """Return (path_voxels, checkpoints, cruise_speed_ms)."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    checkpoints = [Checkpoint(**c) for c in data['checkpoints']]
    return data['path_voxels'], checkpoints, data['cruise_speed_ms']


# ======================================================================
# Demo
# ======================================================================
if __name__ == "__main__":
    import os
    from voxel_grid_builder import VoxelGrid3D   # your existing module

    _DIR = os.path.dirname(os.path.abspath(__file__))
    _DATA_DIR = os.path.join(_DIR, 'data')
    _OUTPUT_DIR = os.path.join(_DIR, 'outputs')
    os.makedirs(_DATA_DIR, exist_ok=True)
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    voxel_grid = VoxelGrid3D.load(os.path.join(_DATA_DIR, 'voxel_grid_westmead_liverpool.pkl'))

    # Hospital endpoints (lat, lon, alt_m). 100 m is mid-band cruise.
    westmead = (-33.8078, 150.9875, 100.0)
    liverpool = (-33.9173, 150.9233, 100.0)

    # --- A* search — band matches AORVAEnv (MIN_ALT_M=25, MAX_ALT_M=150) ---
    pathfinder = AStarPathfinder(voxel_grid, min_alt_m=25.0, max_alt_m=150.0)
    path = pathfinder.find_path(westmead, liverpool)
    if path is None:
        raise RuntimeError("No feasible path found")

    # --- Checkpoints ---
    CRUISE_SPEED = 20.0  # m/s
    checkpoints = compute_checkpoints(
        path, voxel_grid, n_checkpoints=15, cruise_speed_ms=CRUISE_SPEED
    )

    print("\n=== Reference checkpoints ===")
    print(f"{'k':>2}  {'lat':>9}  {'lon':>9}  {'alt':>6}  "
          f"{'dist (m)':>9}  {'t (s)':>7}")
    for c in checkpoints:
        lat, lon, alt = c.latlon_alt
        print(f"{c.index:2d}  {lat:9.4f}  {lon:9.4f}  {alt:6.0f}  "
              f"{c.cumulative_distance_m:9.1f}  {c.target_time_s:7.1f}")

    # --- Save & visualise ---
    save_trajectory(checkpoints, path, CRUISE_SPEED,
                    os.path.join(_DATA_DIR, 'reference_trajectory.pkl'))
    visualize_trajectory(voxel_grid, path, checkpoints,
                         save_path=os.path.join(_OUTPUT_DIR, 'reference_trajectory.png'))

    print("\nDone.")
