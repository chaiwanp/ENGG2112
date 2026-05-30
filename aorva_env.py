"""
aorva_env.py

Gymnasium environment for training PPO and SAC agents on the AORVA
drone organ-delivery task.

Wraps:
  - VoxelGrid3D                (buildings + no-fly zones)
  - WindField3D                (freestream + log-law wind)
  - reference trajectory       (15 checkpoints with target arrival times)
into a Stable-Baselines-3-compatible step/reset interface.

Physics
-------
Lightweight kinematic simulator (first-order velocity dynamics, wind as
a velocity perturbation). Much faster than AirSim and fully adequate
for training. To swap in AirSim for final evaluation, replace the
`_step_physics` method with AirSim client calls -- every other method
stays the same.

Observation space (22 dims, all ~[-1, 1])
-----------------------------------------
    rel_goal (3)          normalised vector drone -> goal (÷ 2× straight-line dist)
    velocity (3)          current velocity / [V_MAX_XY, V_MAX_XY, V_MAX_Z]
    wind_here (3)         wind at drone position / 20 m/s
    wind_ahead (3)        wind sampled 2 s along velocity vector
    battery (1)           [0, 1]
    dist_next_cp (1)      distance to next checkpoint / total path length
    time_dev_cp (1)       (sim_time - target_time) / total_expected_time
    unit_to_cp (3)        unit vector toward next checkpoint
    alt_low_margin (1)    (pos_z - MIN_ALT) / (MAX_ALT - MIN_ALT)  — 0 = at floor
    alt_high_margin (1)   (MAX_ALT - pos_z) / (MAX_ALT - MIN_ALT)  — 0 = at ceiling
    urgency_n (1)         urgency / 1.5  — scales time-deviation weight
    drain_n (1)           battery drain rate this step / MAX_DRAIN_PER_STEP

Action space (3 dims, [-1, 1])
------------------------------
    Desired velocity vector, rescaled internally to +/- V_MAX m/s.

Reward (proposal Eq. 1)
-----------------------
    R = -(w1 * T + w2 * R_k + w3 * E) + sparse bonuses
Weights dynamically scale with organ urgency (w1) and battery state (w3)
per the project proposal.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from voxel_grid_builder import VoxelGrid3D
from wind_field_interpolator import WindField3D
from trajectory_planner import load_trajectory

_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_DIR, 'data')
_DEFAULT_VOXEL_GRID = os.path.join(_DATA_DIR, 'voxel_grid_westmead_liverpool.pkl')
_DEFAULT_TRAJECTORY = os.path.join(_DATA_DIR, 'reference_trajectory.pkl')
_DEFAULT_WIND_DF = os.path.join(_DATA_DIR, 'wind_spatial_real.csv')


# --------- Physics / drone constants ---------
V_MAX_XY = 25.0       # max commanded horizontal velocity (m/s)
V_MAX_Z  = 5.0        # max commanded vertical velocity (m/s) — realistic climb/descent rate
DT = 0.1              # sim timestep (s) -> 10 Hz control
TAU_V = 0.2           # velocity response time constant (s) — faster attitude response
BATTERY_CAPACITY_J = 900_000   # 250 Wh — reasonable medical delivery drone capacity
CRUISE_ALT_M = 100.0  # target cruise altitude (m)
# Fraction of wind the autopilot's attitude controller can actively cancel.
# 0.0 = pure disturbance (old behaviour), 1.0 = full rejection (no wind effect).
# At 0.7 the drone fights 70% of each wind component using extra thrust, which
# drains battery proportionally — making the energy reward term more meaningful.
WIND_REJECTION_COEFF = 0.7
GOAL_RADIUS_M = 30.0
CHECKPOINT_RADIUS_M = 60.0
MIN_ALT_M = 25.0
MAX_ALT_M = 150.0


class AORVAEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 voxel_grid_path: str = _DEFAULT_VOXEL_GRID,
                 trajectory_path: str = _DEFAULT_TRAJECTORY,
                 wind_df_path: str = _DEFAULT_WIND_DF,
                 render_mode: str | None = None):
        super().__init__()

        # --- Load artefacts built in previous stages ---
        self.voxel_grid = VoxelGrid3D.load(voxel_grid_path)

        self.wind_df = pd.read_csv(wind_df_path)
        self.wind_df['timestamp'] = pd.to_datetime(self.wind_df['timestamp'])
        self._spatial_wind = {'lat', 'lon'}.issubset(self.wind_df.columns)
        self._wind_timestamps = (
            self.wind_df['timestamp'].unique() if self._spatial_wind else None
        )
        self.wind_field = WindField3D(
            self.voxel_grid, reference_height_m=10.0, alpha=0.25,
            wind_data_df=self.wind_df if self._spatial_wind else None,
        )
        if self._spatial_wind:
            self.wind_field.interpolate_wind_field()
        else:
            self.wind_field.set_from_dataframe(self.wind_df, timestamp_idx=-1)

        (self.path_voxels,
         self.checkpoints,
         self.cruise_speed) = load_trajectory(trajectory_path)

        # --- Pre-compute world coordinates for start/goal ---
        start_latlon = self.checkpoints[0].latlon_alt
        goal_latlon = self.checkpoints[-1].latlon_alt
        self._start_world = self._latlon_to_world(*start_latlon)
        self._goal_world = self._latlon_to_world(*goal_latlon)
        self._total_straight_dist = float(
            np.linalg.norm(self._goal_world - self._start_world)
        )

        # Target end-to-end flight time from reference trajectory
        self.total_target_time = self.checkpoints[-1].target_time_s
        self.max_episode_time = 2.5 * self.total_target_time

        # --- Spaces ---
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(22,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        self.render_mode = render_mode
        self._trajectory_log: list = []
        self._termination_reason: str | None = None

        # Max battery drain expected per step (W * DT / capacity) — used to normalise drain obs.
        # Derived from: max thrust (~687 W) + max wind rejection (~314 W) at 0.1 s / 900 kJ.
        self._MAX_DRAIN_PER_STEP = 2e-4

        # State populated by reset()
        self.pos = np.zeros(3, dtype=np.float32)
        self.vel = np.zeros(3, dtype=np.float32)
        self.battery = 1.0
        self.sim_time = 0.0
        self.urgency = 1.0
        self._drain_n = 0.0           # normalised battery drain rate from last step
        self.next_checkpoint_idx = 1
        self.checkpoint_deviations: list = []

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------
    def _latlon_to_world(self, lat: float, lon: float, alt_m: float) -> np.ndarray:
        """Local Cartesian frame in metres, origin at voxel grid (min_lon, min_lat)."""
        min_lon, min_lat, _, _ = self.voxel_grid.bounds
        x = (lon - min_lon) * self.voxel_grid.m_per_deg_lon
        y = (lat - min_lat) * self.voxel_grid.m_per_deg_lat
        return np.array([x, y, alt_m], dtype=np.float32)

    def _world_to_voxel(self, pos_world: np.ndarray) -> tuple[int, int, int]:
        vs  = self.voxel_grid.voxel_size_m
        vsz = self.voxel_grid.voxel_size_z_m
        ix = int(np.clip(pos_world[0] / vs,  0, self.voxel_grid.nx - 1))
        iy = int(np.clip(pos_world[1] / vs,  0, self.voxel_grid.ny - 1))
        iz = int(np.clip(pos_world[2] / vsz, 0, self.voxel_grid.nz - 1))
        return ix, iy, iz

    # ------------------------------------------------------------------
    # Gym API: reset
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomise wind by sampling a historical timestamp
        if self._spatial_wind:
            ts_idx = int(self.np_random.integers(0, len(self._wind_timestamps)))
            chosen_ts = pd.Timestamp(self._wind_timestamps[ts_idx])
            self.wind_field.interpolate_wind_field(timestamp=chosen_ts)
        else:
            wind_idx = int(self.np_random.integers(0, len(self.wind_df)))
            self.wind_field.set_from_dataframe(self.wind_df, timestamp_idx=wind_idx)

        # Randomise urgency (scales the time-deviation weight)
        self.urgency = float(self.np_random.uniform(0.8, 1.5))

        # --- Initial state ---
        # We assume takeoff is handled by a low-level autopilot and the RL
        # agent takes over at cruise altitude. Spawn at the start checkpoint
        # (which is already at cruise alt from the A* path) with an initial
        # velocity already pointed at the first downstream checkpoint at
        # ~70% of cruise speed. This avoids the dead-start window where
        # zero velocity + crosswind blows the drone off course before the
        # control loop can build thrust.
        self.pos = self._start_world.copy()

        first_cp_world = self._latlon_to_world(*self.checkpoints[1].latlon_alt)
        direction = first_cp_world - self.pos
        direction /= np.linalg.norm(direction) + 1e-6
        self.vel = (direction * self.cruise_speed * 0.7).astype(np.float32)

        self.battery = 1.0
        self.sim_time = 0.0
        self._drain_n = 0.0
        self.next_checkpoint_idx = 1
        self.checkpoint_deviations = []
        self._trajectory_log = [self.pos.copy()]
        self._prev_dist_to_goal = float(np.linalg.norm(self.pos - self._goal_world))

        return self._get_observation(), self._get_info(episode_done=False)

    # ------------------------------------------------------------------
    # Gym API: step
    # ------------------------------------------------------------------
    def step(self, action):
        desired_vel = np.clip(action, -1.0, 1.0) * np.array([V_MAX_XY, V_MAX_XY, V_MAX_Z], dtype=np.float32)

        battery_before = self.battery
        self._step_physics(desired_vel)
        self._drain_n = float(np.clip(
            (battery_before - self.battery) / self._MAX_DRAIN_PER_STEP, 0.0, 1.0
        ))
        self.sim_time += DT
        self._trajectory_log.append(self.pos.copy())

        # Per-step shaping reward
        reward = self._compute_step_reward(desired_vel)

        # Checkpoint passage reward/penalty
        reward += self._check_checkpoint()

        # Termination
        terminated, term_reward = self._check_termination()
        reward += term_reward
        truncated = self.sim_time >= self.max_episode_time
        if truncated and not terminated:
            reward -= 100.0   # timeout penalty

        episode_done = terminated or truncated
        return (self._get_observation(),
                float(reward),
                bool(terminated),
                bool(truncated),
                self._get_info(episode_done=episode_done))

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------
    def _step_physics(self, desired_vel: np.ndarray) -> None:
        """
        Kinematic drone with first-order velocity dynamics + wind advection.

        This is the only method that would change for AirSim integration.
        For AirSim, replace with: client.simSetWind(...),
        client.moveByVelocityAsync(...), pose = client.simGetVehiclePose().
        """
        ix, iy, iz = self._world_to_voxel(self.pos)
        wind = np.array([
            self.wind_field.u_field[ix, iy, iz],
            self.wind_field.v_field[ix, iy, iz],
            self.wind_field.w_field[ix, iy, iz],
        ], dtype=np.float32)

        # First-order velocity response to command (faster TAU_V = better authority)
        self.vel += (desired_vel - self.vel) * (DT / TAU_V)

        # Partial wind rejection: attitude controller cancels WIND_REJECTION_COEFF of
        # horizontal wind. Vertical wind is nearly fully rejected (stiff altitude loop),
        # meaning updrafts/downdrafts cost thrust but don't move the drone much.
        ALT_HOLD_REJECTION = 0.97   # near-full vertical rejection; residual ≈ 3%
        wind_residual = np.array([
            wind[0] * (1.0 - WIND_REJECTION_COEFF),
            wind[1] * (1.0 - WIND_REJECTION_COEFF),
            wind[2] * (1.0 - ALT_HOLD_REJECTION),
        ], dtype=np.float32)
        wind_cancelled = wind - wind_residual

        effective_vel = self.vel + wind_residual
        self.pos += effective_vel * DT

        # Energy: baseline hover + thrust to track velocity + wind rejection.
        # Autopilot altitude correction removed — altitude management is the agent's
        # responsibility, guided by the cruise altitude penalty and observation margins.
        thrust_power    = float(np.sum(self.vel ** 2)) * 0.5 + 50.0
        rejection_power = float(np.sum(wind_cancelled ** 2)) * 0.8
        self.battery = max(
            0.0,
            self.battery - ((thrust_power + rejection_power) * DT) / BATTERY_CAPACITY_J,
        )

    # ------------------------------------------------------------------
    # Reward components
    # ------------------------------------------------------------------
    def _compute_step_reward(self, desired_vel: np.ndarray) -> float:
        """
        Per-step terms of R = -(w2 * R_k + w3 * E) plus progress shaping.
        The w1 * T term is checkpoint-based; see _check_checkpoint().
        """
        # --- w2 * R_k : ground risk ---
        # Uses ABS 2021 Census population density rasterised into the voxel
        # grid by scripts/02b_download_population.py.  Liverpool LGA is
        # exempted (density zeroed) so the agent isn't penalised for flying
        # toward the delivery destination.  Falls back to an altitude proxy
        # if the density layer hasn't been baked in yet.
        ix, iy, iz = self._world_to_voxel(self.pos)
        w2 = 0.5
        pop_density = getattr(self.voxel_grid, 'population_density', None)
        if pop_density is not None:
            risk = float(np.tanh(pop_density[ix, iy] / 5000.0))
        else:
            risk = max(0.0, 1.0 - iz / self.voxel_grid.nz)
        risk_cost = w2 * risk * 0.02

        # --- w3 * E : energy, weighted up as battery depletes ---
        w3 = 1.0 + (1.0 - self.battery) * 2.0
        # Normalise by max possible |desired_vel|^2 across all axes
        speed_sq_n = float(np.sum(desired_vel ** 2)) / (V_MAX_XY ** 2)
        energy_cost = w3 * speed_sq_n * 0.02

        # --- Delta-progress: reward for getting closer each step ---
        dist_to_goal = float(np.linalg.norm(self.pos - self._goal_world))
        delta_dist = self._prev_dist_to_goal - dist_to_goal  # positive = closer
        self._prev_dist_to_goal = dist_to_goal
        progress_reward = delta_dist * 0.2

        # Soft altitude boundary — symmetric gradient before hard termination
        alt = self.pos[2]
        alt_low_warn  = MIN_ALT_M + 20.0
        alt_high_warn = MAX_ALT_M - 20.0
        if alt < alt_low_warn:
            alt_penalty = ((alt_low_warn - alt) / 20.0) * 0.5
        elif alt > alt_high_warn:
            alt_penalty = ((alt - alt_high_warn) / 20.0) * 0.5
        else:
            alt_penalty = 0.0

        # Cruise altitude cost: continuous pull toward 100 m throughout the flight
        alt_error = abs(self.pos[2] - CRUISE_ALT_M) / 50.0
        altitude_cost = 0.05 * alt_error

        # Vertical velocity cost: penalise unnecessary climb/descent
        vz_cost = 0.02 * abs(self.vel[2]) / V_MAX_Z

        # Small per-step cost to discourage loitering
        step_cost = 0.01

        return -(risk_cost + energy_cost + alt_penalty + altitude_cost + vz_cost + step_cost) + progress_reward

    def _check_checkpoint(self) -> float:
        """w1 * T penalty triggered when drone passes a checkpoint."""
        if self.next_checkpoint_idx >= len(self.checkpoints):
            return 0.0

        cp = self.checkpoints[self.next_checkpoint_idx]
        cp_world = self._latlon_to_world(*cp.latlon_alt)
        dist = float(np.linalg.norm(self.pos - cp_world))

        if dist < CHECKPOINT_RADIUS_M:
            deviation = self.sim_time - cp.target_time_s
            self.checkpoint_deviations.append(deviation)
            self.next_checkpoint_idx += 1
            # Small positive reward for reaching the checkpoint — gives the agent
            # an incentive to follow the planned path, not just aim at the goal.
            # Lateness penalty on top: only penalise being late, not early.
            lateness = max(0.0, deviation)
            timing_penalty = -min(self.urgency * lateness * 0.3, 30.0)
            return 20.0 + timing_penalty

        return 0.0

    def _check_termination(self) -> tuple[bool, float]:
        """Return (done, reward_adjustment). Sets self._termination_reason."""
        # Goal reached
        if np.linalg.norm(self.pos - self._goal_world) < GOAL_RADIUS_M:
            total_dev = sum(abs(d) for d in self.checkpoint_deviations)
            accuracy_bonus = 500.0 / (1.0 + total_dev / 10.0)
            battery_bonus = self.battery * 100.0
            self._termination_reason = "GOAL REACHED"
            return True, 1000.0 + accuracy_bonus + battery_bonus

        # Collision with building or no-fly zone
        ix, iy, iz = self._world_to_voxel(self.pos)
        if self.voxel_grid.grid[ix, iy, iz] == 1:
            self._termination_reason = f"COLLISION (voxel [{ix},{iy},{iz}] alt={self.pos[2]:.1f}m)"
            return True, -500.0

        # Altitude violations (safety)
        if self.pos[2] < MIN_ALT_M:
            self._termination_reason = f"TOO LOW (alt={self.pos[2]:.1f}m < {MIN_ALT_M}m)"
            return True, -300.0
        if self.pos[2] > MAX_ALT_M:
            self._termination_reason = f"TOO HIGH (alt={self.pos[2]:.1f}m > {MAX_ALT_M}m)"
            return True, -300.0

        # Out of horizontal bounds
        if (self.pos[0] < 0 or self.pos[0] > self.voxel_grid.width_m or
                self.pos[1] < 0 or self.pos[1] > self.voxel_grid.length_m):
            self._termination_reason = f"OUT OF BOUNDS (pos=[{self.pos[0]:.0f},{self.pos[1]:.0f}])"
            return True, -200.0

        # Battery depleted
        if self.battery <= 0.0:
            self._termination_reason = "BATTERY DEAD"
            return True, -400.0

        self._termination_reason = None
        return False, 0.0

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _get_observation(self) -> np.ndarray:
        # Divide by 2× straight-line distance so values stay in [-1,1] even when the
        # drone wanders off-corridor; avoids silent clipping that hides true deviation.
        rel_goal = (self._goal_world - self.pos) / (2.0 * self._total_straight_dist)
        vel_n = self.vel / np.array([V_MAX_XY, V_MAX_XY, V_MAX_Z], dtype=np.float32)

        ix, iy, iz = self._world_to_voxel(self.pos)
        wind_here = np.array([
            self.wind_field.u_field[ix, iy, iz],
            self.wind_field.v_field[ix, iy, iz],
            self.wind_field.w_field[ix, iy, iz],
        ], dtype=np.float32) / 20.0

        # 2 s lookahead along current velocity
        lookahead = self.pos + self.vel * 2.0
        lx, ly, lz = self._world_to_voxel(lookahead)
        wind_ahead = np.array([
            self.wind_field.u_field[lx, ly, lz],
            self.wind_field.v_field[lx, ly, lz],
            self.wind_field.w_field[lx, ly, lz],
        ], dtype=np.float32) / 20.0

        if self.next_checkpoint_idx < len(self.checkpoints):
            cp = self.checkpoints[self.next_checkpoint_idx]
            cp_world = self._latlon_to_world(*cp.latlon_alt)
            to_cp = cp_world - self.pos
            dist_cp = float(np.linalg.norm(to_cp))
            unit_to_cp = to_cp / (dist_cp + 1e-6)
            time_dev = (self.sim_time - cp.target_time_s) / self.total_target_time
            dist_cp_n = dist_cp / self._total_straight_dist
        else:
            direction = self._goal_world - self.pos
            unit_to_cp = direction / (np.linalg.norm(direction) + 1e-6)
            time_dev = 0.0
            dist_cp_n = 0.0

        alt_range = MAX_ALT_M - MIN_ALT_M
        alt_low_margin  = (self.pos[2] - MIN_ALT_M) / alt_range   # 0 = at floor
        alt_high_margin = (MAX_ALT_M - self.pos[2]) / alt_range   # 0 = at ceiling

        urgency_n = self.urgency / 1.5   # normalised to ~[0.53, 1.0]

        obs = np.concatenate([
            rel_goal,                              # 3
            vel_n,                                 # 3
            wind_here,                             # 3
            wind_ahead,                            # 3
            [self.battery],                        # 1
            [dist_cp_n],                           # 1
            [np.clip(time_dev, -1.0, 1.0)],        # 1
            unit_to_cp,                            # 3
            [alt_low_margin],                      # 1  (how far above floor)
            [alt_high_margin],                     # 1  (how far below ceiling)
            [urgency_n],                           # 1  (delivery urgency)
            [self._drain_n],                       # 1  (battery drain rate)
        ]).astype(np.float32)

        return np.clip(obs, -1.0, 1.0)

    def _get_info(self, episode_done: bool = False) -> dict:
        # Trajectory is only included at episode end. During training steps it
        # would be pickled through SubprocVecEnv pipes on every step, growing
        # O(n) per step and causing O(n²) total IPC overhead per episode.
        return {
            'sim_time': self.sim_time,
            'battery': self.battery,
            'urgency': self.urgency,
            'checkpoints_passed': self.next_checkpoint_idx - 1,
            'checkpoint_deviations': self.checkpoint_deviations.copy(),
            'position': self.pos.copy(),
            'trajectory': [p.copy() for p in self._trajectory_log] if episode_done else [],
        }

    def render(self):
        # Visualisation handled post-hoc by evaluate_agents.py
        pass


# ======================================================================
# Smoke test  —  run with:  python aorva_env.py
# ======================================================================
if __name__ == "__main__":
    N_EPISODES   = 3     # how many episodes to run
    LOG_EVERY    = 200   # print a status line every this many steps

    env = AORVAEnv()

    # ---- Voxel grid sanity -----------------------------------------------
    vg = env.voxel_grid
    print("=" * 60)
    print("VOXEL GRID")
    print(f"  XY voxel size : {vg.voxel_size_m} m")
    print(f"  Z  voxel size : {vg.voxel_size_z_m} m")
    print(f"  Grid shape    : {vg.nx} x {vg.ny} x {vg.nz}")
    print(f"  World size    : {vg.width_m:.0f} m x {vg.length_m:.0f} m x {vg.max_height_m:.0f} m")
    print(f"  Altitude band : {MIN_ALT_M}–{MAX_ALT_M} m  "
          f"(z_index {int(MIN_ALT_M/vg.voxel_size_z_m)}–{int(MAX_ALT_M/vg.voxel_size_z_m)})")

    # ---- Trajectory sanity -----------------------------------------------
    print("\nTRAJECTORY")
    print(f"  Checkpoints   : {len(env.checkpoints)}")
    print(f"  Total time    : {env.total_target_time:.1f} s  "
          f"(max episode: {env.max_episode_time:.1f} s)")
    start_vox = env._world_to_voxel(env._start_world)
    goal_vox  = env._world_to_voxel(env._goal_world)
    print(f"  Start world   : {env._start_world}  -> voxel {start_vox}  "
          f"(z_index {start_vox[2]} = {start_vox[2]*vg.voxel_size_z_m:.0f} m alt)")
    print(f"  Goal  world   : {env._goal_world}  -> voxel {goal_vox}  "
          f"(z_index {goal_vox[2]} = {goal_vox[2]*vg.voxel_size_z_m:.0f} m alt)")
    print(f"  Start voxel occupied? {bool(vg.grid[start_vox])}")
    print(f"  Goal  voxel occupied? {bool(vg.grid[goal_vox])}")
    print("=" * 60)

    # ---- Episode loop ----------------------------------------------------
    for ep in range(N_EPISODES):
        obs, info = env.reset(seed=ep)

        # Verify reset position immediately
        ix, iy, iz = env._world_to_voxel(env.pos)
        print(f"\n[EP {ep+1}] Reset: pos={env.pos}  "
              f"voxel=[{ix},{iy},{iz}]  alt={env.pos[2]:.1f}m  "
              f"occupied={bool(vg.grid[ix, iy, iz])}")
        print(f"         obs range [{obs.min():.3f}, {obs.max():.3f}]")

        total_reward = 0.0
        for t in range(10_000):
            # Naive policy: fly straight toward goal
            direction = env._goal_world - env.pos
            direction /= np.linalg.norm(direction) + 1e-6
            action = direction.astype(np.float32)

            obs, r, terminated, truncated, info = env.step(action)
            total_reward += r

            # Periodic status line
            if (t + 1) % LOG_EVERY == 0:
                ix, iy, iz = env._world_to_voxel(env.pos)
                dist_to_goal = np.linalg.norm(env.pos - env._goal_world)
                print(f"  step {t+1:5d} | alt={env.pos[2]:6.1f}m  z_idx={iz:2d}  "
                      f"dist_goal={dist_to_goal:7.0f}m  "
                      f"battery={env.battery:.2f}  "
                      f"cp={info['checkpoints_passed']}/{len(env.checkpoints)-1}  "
                      f"reward={total_reward:8.1f}")

            if terminated or truncated:
                reason = getattr(env, '_termination_reason', 'TIMEOUT') or 'TIMEOUT'
                print(f"\n  --> Episode ended at step {t+1}  ({reason})")
                print(f"      total_reward={total_reward:.1f}  "
                      f"sim_time={info['sim_time']:.1f}s  "
                      f"checkpoints={info['checkpoints_passed']}/{len(env.checkpoints)-1}  "
                      f"battery={info['battery']:.2f}")
                break

    print("\nSmoke test complete.")

        