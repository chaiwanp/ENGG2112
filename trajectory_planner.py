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
_NEIGHBOR_COSTS: List[float] = [
    float(np.sqrt(dx * dx + dy * dy + dz * dz))
    for (dx, dy, dz) in _NEIGHBOR_OFFSETS
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

    def __init__(self, voxel_grid, min_alt_m: float = 50.0,
                 max_alt_m: float = 150.0):
        self.voxel_grid = voxel_grid
        self.grid = voxel_grid.grid
        self.vs = voxel_grid.voxel_size_m
        self.nx, self.ny, self.nz = voxel_grid.nx, voxel_grid.ny, voxel_grid.nz

        self.min_z = max(0, int(min_alt_m / self.vs))
        self.max_z = min(self.nz - 1, int(max_alt_m / self.vs))

        if self.min_z > self.max_z:
            raise ValueError(
                f"min_alt {min_alt_m} m exceeds max_alt {max_alt_m} m for this grid"
            )

    # --- traversability helpers -------------------------------------------
    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.nx and 0 <= y < self.ny

    def _traversable(self, x: int, y: int, z: int) -> bool:
        if not self._in_bounds(x, y):
            return False
        if z < self.min_z or z > self.max_z:
            return False
        return self.grid[x, y, z] == 0

    @staticmethod
    def _heuristic(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]
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

                tentative_g = g + _NEIGHBOR_COSTS[i]
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
    pts = np.asarray(path_voxels, dtype=np.float64)

    # Cumulative arc length in metres
    seg_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1) * vs
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
    """
    Three-panel figure: top-down map, altitude profile, and checkpoint timeline.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.collections import LineCollection

    path = np.asarray(path_voxels)
    vs   = voxel_grid.voxel_size_m

    # Arc-length parameterisation for profile/timeline panels
    pts         = path.astype(np.float64)
    seg_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1) * vs
    cumulative  = np.concatenate(([0.0], np.cumsum(seg_lengths)))
    alts        = path[:, 2] * vs
    total_dist  = float(cumulative[-1])
    total_time  = checkpoints[-1].target_time_s

    cp_voxels = np.array([c.voxel for c in checkpoints])
    cp_dist   = [c.cumulative_distance_m for c in checkpoints]
    cp_alt    = [c.latlon_alt[2] for c in checkpoints]
    cp_time   = [c.target_time_s for c in checkpoints]

    fig = plt.figure(figsize=(20, 13))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28)
    ax_top  = fig.add_subplot(gs[0, :])   # top-down spanning both columns
    ax_side = fig.add_subplot(gs[1, 0])   # altitude profile
    ax_time = fig.add_subplot(gs[1, 1])   # checkpoint timing

    # -- Top-down map --------------------------------------------------
    max_z_idx   = int(min(path[:, 2].max() + 1, voxel_grid.nz))
    occ_proj    = np.max(voxel_grid.grid[:, :, :max_z_idx], axis=2).T
    nfz_proj    = np.max(
        (voxel_grid.metadata[:, :, :max_z_idx] == 2).astype(np.uint8),
        axis=2
    ).T

    ax_top.imshow(occ_proj, origin='lower', cmap='Greys', alpha=0.35,
                  extent=[0, voxel_grid.nx, 0, voxel_grid.ny], aspect='auto')
    ax_top.imshow(
        np.ma.masked_where(nfz_proj == 0, nfz_proj.astype(float)),
        origin='lower', cmap='Oranges', alpha=0.55, vmin=0, vmax=1,
        extent=[0, voxel_grid.nx, 0, voxel_grid.ny], aspect='auto',
    )

    # Colour path segments by altitude
    points = path[:, :2].astype(float)[:, np.newaxis, :]
    segs   = np.concatenate([points[:-1], points[1:]], axis=1)
    lc     = LineCollection(segs, array=alts[:-1], cmap='plasma',
                             linewidth=2.0, zorder=4)
    ax_top.add_collection(lc)
    cb = plt.colorbar(lc, ax=ax_top, fraction=0.018, pad=0.01)
    cb.set_label('Altitude (m)', fontsize=8)

    # Checkpoint markers with index labels
    ax_top.scatter(cp_voxels[:, 0], cp_voxels[:, 1],
                   c='red', s=70, zorder=6, label='Checkpoints',
                   edgecolors='white', linewidths=0.8)
    for c in checkpoints:
        ax_top.annotate(
            str(c.index),
            (c.voxel[0], c.voxel[1]),
            fontsize=7, ha='center', va='center', color='white',
            fontweight='bold', zorder=7,
        )

    # Hospital markers
    ax_top.plot(path[0, 0], path[0, 1], '*', color='dodgerblue',
                markersize=24, markeredgecolor='white', markeredgewidth=1.5,
                label='Westmead Hospital', zorder=8)
    ax_top.plot(path[-1, 0], path[-1, 1], '*', color='limegreen',
                markersize=24, markeredgecolor='black', markeredgewidth=0.8,
                label='Liverpool Hospital', zorder=8)

    ax_top.set_xlabel('Grid X  (west -> east)', fontsize=9)
    ax_top.set_ylabel('Grid Y  (south -> north)', fontsize=9)
    ax_top.set_title(
        f'A* Reference Path - Top-down View\n'
        f'Total distance: {total_dist/1000:.2f} km  |  '
        f'Zero-wind flight time: {total_time:.0f} s ({total_time/60:.1f} min)  |  '
        f'Cruise speed: 20 m/s  |  {len(checkpoints)} checkpoints',
        fontsize=11,
    )
    ax_top.legend(loc='upper right', fontsize=9, framealpha=0.92)
    ax_top.set_xlim(0, voxel_grid.nx)
    ax_top.set_ylim(0, voxel_grid.ny)

    # -- Altitude profile ---------------------------------------------
    ax_side.fill_between(cumulative / 1000, alts, alpha=0.18, color='steelblue')
    ax_side.plot(cumulative / 1000, alts, '-', color='steelblue',
                 linewidth=1.8, label='A* path altitude')
    ax_side.axhline(50,  color='tomato', linestyle=':', linewidth=1.2,
                    label='Min alt 50 m', alpha=0.8)
    ax_side.axhline(150, color='tomato', linestyle=':', linewidth=1.2,
                    label='Max alt 150 m', alpha=0.8)
    ax_side.scatter([d / 1000 for d in cp_dist], cp_alt,
                    c='red', s=60, zorder=6, label='Checkpoints',
                    edgecolors='white', linewidths=0.8)

    for c in checkpoints:
        ax_side.annotate(
            str(c.index),
            (c.cumulative_distance_m / 1000, c.latlon_alt[2]),
            textcoords='offset points', xytext=(0, 8),
            fontsize=7, ha='center', color='darkred',
        )

    ax_side.set_xlabel('Cumulative distance (km)', fontsize=9)
    ax_side.set_ylabel('Altitude (m)', fontsize=9)
    ax_side.set_title('Altitude Profile', fontsize=10)
    ax_side.set_ylim(0, voxel_grid.voxel_size_m * voxel_grid.nz * 0.5)
    ax_side.legend(fontsize=8, loc='upper right')
    ax_side.grid(True, alpha=0.3)

    # -- Checkpoint timing --------------------------------------------
    ax_time.plot(range(len(checkpoints)), cp_time, 'o-',
                 color='steelblue', linewidth=2, markersize=7,
                 label='Target time (s)')
    ax_time.fill_between(range(len(checkpoints)), cp_time, alpha=0.15,
                         color='steelblue')

    for i, (t, d) in enumerate(zip(cp_time, cp_dist)):
        ax_time.annotate(
            f'{t:.0f}s\n{d/1000:.1f}km',
            (i, t), textcoords='offset points', xytext=(4, 4),
            fontsize=6.5, color='navy',
        )

    ax_time.set_xlabel('Checkpoint index', fontsize=9)
    ax_time.set_ylabel('Target arrival time (s)', fontsize=9)
    ax_time.set_title(
        'Checkpoint Target Timeline\n'
        '(Zero-wind schedule - RL agent evaluated against these)',
        fontsize=10,
    )
    ax_time.set_xticks(range(len(checkpoints)))
    ax_time.grid(True, alpha=0.3)

    # Second y-axis in minutes
    ax_time2 = ax_time.twinx()
    ax_time2.set_ylim(np.array(ax_time.get_ylim()) / 60)
    ax_time2.set_ylabel('Time (min)', fontsize=8)

    plt.suptitle('Phase 2 - A* Reference Trajectory with 15 Checkpoints',
                 fontsize=13, fontweight='bold')

    if save_path:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=250, bbox_inches='tight')
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

    os.makedirs('data', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    voxel_grid = VoxelGrid3D.load('data/voxel_grid_westmead_liverpool.pkl')

    # Hospital endpoints (lat, lon, alt_m). 100 m is mid-band cruise.
    westmead = (-33.8078, 150.9875, 100.0)
    liverpool = (-33.9173, 150.9233, 100.0)

    # --- A* search ---
    pathfinder = AStarPathfinder(voxel_grid, min_alt_m=50.0, max_alt_m=150.0)
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
                    'data/reference_trajectory.pkl')
    visualize_trajectory(voxel_grid, path, checkpoints,
                         save_path='outputs/reference_trajectory.png')

    print("\nDone.")
