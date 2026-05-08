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

Observation space (18 dims, all ~[-1, 1])
-----------------------------------------
    rel_goal (3)       normalised vector drone -> goal
    velocity (3)       current velocity / V_MAX
    wind_here (3)      wind at drone position / 20 m/s
    wind_ahead (3)     wind sampled 2 s along velocity vector
    battery (1)        [0, 1]
    dist_next_cp (1)   distance to next checkpoint / total path length
    time_dev_cp (1)    (sim_time - target_time) / total_expected_time
    unit_to_cp (3)     unit vector toward next checkpoint

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

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from voxel_grid_builder import VoxelGrid3D
from wind_field_interpolator import WindField3D
from trajectory_planner import load_trajectory



# --------- Physics / drone constants ---------
V_MAX = 25.0          # max commanded velocity (m/s)
DT = 0.1              # sim timestep (s) -> 10 Hz control
TAU_V = 0.3           # velocity response time constant (s)
BATTERY_CAPACITY_J = 300_000   # ~3 kWh; typical medical delivery drone
GOAL_RADIUS_M = 30.0
CHECKPOINT_RADIUS_M = 60.0
MIN_ALT_M = 30.0
MAX_ALT_M = 300.0


class AORVAEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 voxel_grid_path: str = 'data/voxel_grid_westmead_liverpool.pkl',
                 trajectory_path: str = 'data/reference_trajectory.pkl',
                 wind_df_path: str = 'data/wind_historical_real.csv',
                 render_mode: str | None = None):
        super().__init__()

        # --- Load artefacts built in previous stages ---
        self.voxel_grid = VoxelGrid3D.load(voxel_grid_path)

        self.wind_df = pd.read_csv(wind_df_path)
        self.wind_df['timestamp'] = pd.to_datetime(self.wind_df['timestamp'])
        self.wind_field = WindField3D(self.voxel_grid, self.wind_df)
        self.wind_field.interpolate_wind_field()  # initial fill

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
            low=-1.0, high=1.0, shape=(18,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        self.render_mode = render_mode
        self._trajectory_log: list = []

        # State populated by reset()
        self.pos = np.zeros(3, dtype=np.float32)
        self.vel = np.zeros(3, dtype=np.float32)
        self.battery = 1.0
        self.sim_time = 0.0
        self.urgency = 1.0
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
        vs = self.voxel_grid.voxel_size_m
        ix = int(np.clip(pos_world[0] / vs, 0, self.voxel_grid.nx - 1))
        iy = int(np.clip(pos_world[1] / vs, 0, self.voxel_grid.ny - 1))
        iz = int(np.clip(pos_world[2] / vs, 0, self.voxel_grid.nz - 1))
        return ix, iy, iz

    # ------------------------------------------------------------------
    # Gym API: reset
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomise wind by sampling a historical timestamp
        wind_idx = int(self.np_random.integers(0, len(self.wind_df)))
        self.wind_field.interpolate_wind_field(
            timestamp=self.wind_df['timestamp'].iloc[wind_idx]
        )

        # Randomise urgency (scales the time-deviation weight)
        self.urgency = float(self.np_random.uniform(0.8, 1.5))

        # Initial state
        self.pos = self._start_world.copy()
        self.vel = np.zeros(3, dtype=np.float32)
        self.battery = 1.0
        self.sim_time = 0.0
        self.next_checkpoint_idx = 1
        self.checkpoint_deviations = []
        self._trajectory_log = [self.pos.copy()]

        return self._get_observation(), self._get_info()

    # ------------------------------------------------------------------
    # Gym API: step
    # ------------------------------------------------------------------
    def step(self, action):
        desired_vel = np.clip(action, -1.0, 1.0) * V_MAX

        self._step_physics(desired_vel)
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

        return (self._get_observation(),
                float(reward),
                bool(terminated),
                bool(truncated),
                self._get_info())

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
        u, v, w = self.wind_field.get_wind_at_position(ix, iy, iz)
        wind = np.array([u, v, w], dtype=np.float32)

        # First-order velocity response to command
        self.vel += (desired_vel - self.vel) * (DT / TAU_V)

        # Wind advects the drone on top of its commanded velocity
        effective_vel = self.vel + wind
        self.pos += effective_vel * DT

        # Energy drain: baseline hover + thrust^2 proxy
        thrust_power = float(np.sum(desired_vel ** 2)) * 0.5 + 50.0
        self.battery = max(0.0, self.battery - (thrust_power * DT) / BATTERY_CAPACITY_J)

    # ------------------------------------------------------------------
    # Reward components
    # ------------------------------------------------------------------
    def _compute_step_reward(self, desired_vel: np.ndarray) -> float:
        """
        Per-step terms of R = -(w2 * R_k + w3 * E) plus progress shaping.
        The w1 * T term is checkpoint-based; see _check_checkpoint().
        """
        # --- w2 * R_k : ground risk (ABS population density) ---
        # W(rho) = tanh(rho / rho_ref) maps density -> [0,1] risk weight.
        # Altitude discount halves the weight every ALT_HALF metres, reflecting
        # reduced crash-zone footprint and noise impact at higher altitude.
        # See download_abs_population.py for full justification.
        ALT_HALF = 100.0   # metres - risk halves at this altitude
        ix, iy, _ = self._world_to_voxel(self.pos)
        density  = float(self.voxel_grid.density_map[ix, iy])
        w_density = float(np.tanh(density / 5_000.0))
        alt_discount = 1.0 / (1.0 + self.pos[2] / ALT_HALF)
        w2 = 0.5
        risk_cost = w2 * w_density * alt_discount * 0.02

        # --- w3 * E : energy, weighted up as battery depletes ---
        w3 = 1.0 + (1.0 - self.battery) * 2.0
        speed_sq_n = float(np.sum(desired_vel ** 2)) / (V_MAX ** 2)
        energy_cost = w3 * speed_sq_n * 0.02

        # --- Progress toward goal (shaping, helps early exploration) ---
        dist_to_goal = float(np.linalg.norm(self.pos - self._goal_world))
        progress = 1.0 - dist_to_goal / self._total_straight_dist
        progress_reward = progress * 0.05

        # Small per-step cost to discourage loitering
        step_cost = 0.1

        return -(risk_cost + energy_cost + step_cost) + progress_reward

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
            # Linear penalty in |deviation|, scaled by urgency
            return -self.urgency * abs(deviation) * 2.0

        return 0.0

    def _check_termination(self) -> tuple[bool, float]:
        """Return (done, reward_adjustment)."""
        # Goal reached
        if np.linalg.norm(self.pos - self._goal_world) < GOAL_RADIUS_M:
            total_dev = sum(abs(d) for d in self.checkpoint_deviations)
            accuracy_bonus = 500.0 / (1.0 + total_dev / 10.0)
            battery_bonus = self.battery * 100.0
            return True, 1000.0 + accuracy_bonus + battery_bonus

        # Collision with building or no-fly zone
        ix, iy, iz = self._world_to_voxel(self.pos)
        if self.voxel_grid.grid[ix, iy, iz] == 1:
            return True, -500.0

        # Altitude violations (safety)
        if self.pos[2] < MIN_ALT_M:
            return True, -300.0
        if self.pos[2] > MAX_ALT_M:
            return True, -100.0

        # Out of horizontal bounds
        if (self.pos[0] < 0 or self.pos[0] > self.voxel_grid.width_m or
                self.pos[1] < 0 or self.pos[1] > self.voxel_grid.length_m):
            return True, -200.0

        # Battery depleted
        if self.battery <= 0.0:
            return True, -400.0

        return False, 0.0

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _get_observation(self) -> np.ndarray:
        rel_goal = (self._goal_world - self.pos) / self._total_straight_dist
        vel_n = self.vel / V_MAX

        ix, iy, iz = self._world_to_voxel(self.pos)
        u, v, w = self.wind_field.get_wind_at_position(ix, iy, iz)
        wind_here = np.array([u, v, w], dtype=np.float32) / 20.0

        # 2 s lookahead along current velocity
        lookahead = self.pos + self.vel * 2.0
        lx, ly, lz = self._world_to_voxel(lookahead)
        ua, va, wa = self.wind_field.get_wind_at_position(lx, ly, lz)
        wind_ahead = np.array([ua, va, wa], dtype=np.float32) / 20.0

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

        obs = np.concatenate([
            rel_goal,                              # 3
            vel_n,                                 # 3
            wind_here,                             # 3
            wind_ahead,                            # 3
            [self.battery],                        # 1
            [dist_cp_n],                           # 1
            [np.clip(time_dev, -1.0, 1.0)],        # 1
            unit_to_cp,                            # 3
        ]).astype(np.float32)

        return np.clip(obs, -1.0, 1.0)

    def _get_info(self) -> dict:
        return {
            'sim_time': self.sim_time,
            'battery': self.battery,
            'urgency': self.urgency,
            'checkpoints_passed': self.next_checkpoint_idx - 1,
            'checkpoint_deviations': self.checkpoint_deviations.copy(),
            'position': self.pos.copy(),
            'trajectory': [p.copy() for p in self._trajectory_log],
        }

    def render(self):
        # Visualisation handled post-hoc by evaluate_agents.py
        pass


# ======================================================================
# Smoke test
# ======================================================================
if __name__ == "__main__":
    env = AORVAEnv()
    obs, info = env.reset(seed=0)
    print(f"obs shape: {obs.shape}   obs range: [{obs.min():.2f}, {obs.max():.2f}]")
    print(f"action space: {env.action_space}")

    total_reward = 0.0
    for t in range(10_000):
        # Naive policy: fly straight toward goal
        direction = env._goal_world - env.pos
        direction /= np.linalg.norm(direction) + 1e-6
        action = direction
        obs, r, terminated, truncated, info = env.step(action)
        total_reward += r
        if terminated or truncated:
            print(f"Episode ended at step {t}: total reward {total_reward:.1f}")
            print(f"  sim_time = {info['sim_time']:.1f} s")
            print(f"  checkpoints passed: {info['checkpoints_passed']} / "
                  f"{len(env.checkpoints) - 1}")
            print(f"  terminated={terminated}, truncated={truncated}")
            break
