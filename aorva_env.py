"""
aorva_env.py  (v3 - suicide prevention fix)

v2 still had the agent terminating early because the reward landscape
made dying preferable to a long, uncertain flight. Specifically:
  - Progress reward was symmetric: backward motion cost as much as
    forward motion gained. A flailing agent net-zeroed on progress
    while still eating per-step costs.
  - Per-step cost accumulated faster than crash penalty, so dying
    was the optimal strategy during exploration.
  - Crash penalty wasn't large enough to be obviously worse than a
    long flight.

Changes vs v2
-------------
1. Progress reward is now ASYMMETRIC: full reward for forward motion,
   only 10% penalty for backward motion. Flailing is now reward-neutral.
2. Per-step time cost REMOVED. The checkpoint time-deviation penalty
   already encodes time pressure. A flat per-step cost just bleeds.
3. Crash penalty increased to -5000. Survival is now unambiguously
   better than termination.
4. Timeout penalty reduced to -100 (was -300). Timing out should be
   slightly worse than barely failing, but nowhere near as bad as
   crashing.
5. Energy and risk costs reduced further -- they should be tiebreakers,
   not primary signals.
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
V_MAX = 25.0
DT = 0.1
TAU_V = 0.3
BATTERY_CAPACITY_J = 300_000
GOAL_RADIUS_M = 30.0
CHECKPOINT_RADIUS_M = 60.0
MIN_ALT_M = 30.0
MAX_ALT_M = 300.0

# --------- Reward weights (v3 - asymmetric progress) ---------
W_PROGRESS_FORWARD  = 1.0      # full reward for closing on goal
W_PROGRESS_BACKWARD = 0.1      # only 10% penalty for moving away
W_RISK              = 0.002    # tiebreaker only
W_ENERGY            = 0.002    # tiebreaker only
# W_STEP_COST removed entirely - it was the suicide incentive

W_TIME_PENALTY      = 0.5
TIME_TOLERANCE_S    = 15.0
TIME_PENALTY_CAP    = 30.0

# Terminal rewards - crash MUST be unambiguously the worst outcome
REWARD_GOAL              = 3000.0
REWARD_CRASH             = -5000.0   # was -2000
REWARD_LOW_ALT           = -3000.0   # was -1500
REWARD_HIGH_ALT          = -1000.0   # was -500
REWARD_OUT_OF_BOUNDS     = -2000.0   # was -1000
REWARD_BATTERY_DEAD      = -2000.0   # was -1500
REWARD_TIMEOUT           = -100.0    # was -300, now small enough that
                                     # timing out is preferable to crashing


class AORVAEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 voxel_grid_path: str = 'data/voxel_grid_westmead_liverpool.pkl',
                 trajectory_path: str = 'data/reference_trajectory.pkl',
                 wind_df_path: str = 'data/wind_historical_synthetic.csv',
                 render_mode: str | None = None):
        super().__init__()

        self.voxel_grid = VoxelGrid3D.load(voxel_grid_path)

        self.wind_df = pd.read_csv(wind_df_path)
        self.wind_df['timestamp'] = pd.to_datetime(self.wind_df['timestamp'])
        self.wind_field = WindField3D(self.voxel_grid, self.wind_df)
        self.wind_field.interpolate_wind_field()

        (self.path_voxels,
         self.checkpoints,
         self.cruise_speed) = load_trajectory(trajectory_path)

        start_latlon = self.checkpoints[0].latlon_alt
        goal_latlon = self.checkpoints[-1].latlon_alt
        self._start_world = self._latlon_to_world(*start_latlon)
        self._goal_world = self._latlon_to_world(*goal_latlon)
        self._total_straight_dist = float(
            np.linalg.norm(self._goal_world - self._start_world)
        )

        self.total_target_time = self.checkpoints[-1].target_time_s
        self.max_episode_time = 2.5 * self.total_target_time

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(18,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        self.render_mode = render_mode
        self._trajectory_log: list = []

        self.pos = np.zeros(3, dtype=np.float32)
        self.vel = np.zeros(3, dtype=np.float32)
        self.battery = 1.0
        self.sim_time = 0.0
        self.urgency = 1.0
        self.next_checkpoint_idx = 1
        self.checkpoint_deviations: list = []
        self._prev_dist_to_goal = self._total_straight_dist

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------
    def _latlon_to_world(self, lat: float, lon: float, alt_m: float) -> np.ndarray:
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

        wind_idx = int(self.np_random.integers(0, len(self.wind_df)))
        self.wind_field.interpolate_wind_field(
            timestamp=self.wind_df['timestamp'].iloc[wind_idx]
        )

        self.urgency = float(self.np_random.uniform(0.8, 1.5))

        self.pos = self._start_world.copy()
        self.vel = np.zeros(3, dtype=np.float32)
        self.battery = 1.0
        self.sim_time = 0.0
        self.next_checkpoint_idx = 1
        self.checkpoint_deviations = []
        self._trajectory_log = [self.pos.copy()]
        self._prev_dist_to_goal = float(
            np.linalg.norm(self.pos - self._goal_world)
        )

        return self._get_observation(), self._get_info()

    # ------------------------------------------------------------------
    # Gym API: step
    # ------------------------------------------------------------------
    def step(self, action):
        desired_vel = np.clip(action, -1.0, 1.0) * V_MAX

        self._step_physics(desired_vel)
        self.sim_time += DT
        self._trajectory_log.append(self.pos.copy())

        reward = self._compute_step_reward(desired_vel)
        reward += self._check_checkpoint()

        terminated, term_reward = self._check_termination()
        reward += term_reward
        truncated = self.sim_time >= self.max_episode_time
        if truncated and not terminated:
            reward += REWARD_TIMEOUT

        return (self._get_observation(),
                float(reward),
                bool(terminated),
                bool(truncated),
                self._get_info())

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------
    def _step_physics(self, desired_vel: np.ndarray) -> None:
        ix, iy, iz = self._world_to_voxel(self.pos)
        u, v, w = self.wind_field.get_wind_at_position(ix, iy, iz)
        wind = np.array([u, v, w], dtype=np.float32)

        self.vel += (desired_vel - self.vel) * (DT / TAU_V)
        effective_vel = self.vel + wind
        self.pos += effective_vel * DT

        thrust_power = float(np.sum(desired_vel ** 2)) * 0.5 + 50.0
        self.battery = max(0.0, self.battery - (thrust_power * DT) / BATTERY_CAPACITY_J)

    # ------------------------------------------------------------------
    # Reward components
    # ------------------------------------------------------------------
    def _compute_step_reward(self, desired_vel: np.ndarray) -> float:
        """
        v3: asymmetric progress + no per-step bleed.

        Forward motion (delta_dist > 0): full reward
        Backward motion (delta_dist < 0): 10% penalty only

        This means flailing is reward-neutral, which removes the
        incentive to terminate early.
        """
        # --- Asymmetric progress reward ---
        dist_now = float(np.linalg.norm(self.pos - self._goal_world))
        delta_dist = self._prev_dist_to_goal - dist_now
        self._prev_dist_to_goal = dist_now

        if delta_dist > 0:
            progress_reward = W_PROGRESS_FORWARD * delta_dist
        else:
            progress_reward = W_PROGRESS_BACKWARD * delta_dist

        # --- Tiebreaker: ground-risk cost ---
        _, _, iz = self._world_to_voxel(self.pos)
        risk = max(0.0, 1.0 - iz / self.voxel_grid.nz)
        risk_cost = W_RISK * risk

        # --- Tiebreaker: energy cost ---
        battery_factor = 1.0 + (1.0 - self.battery) * 2.0
        speed_sq_n = float(np.sum(desired_vel ** 2)) / (V_MAX ** 2)
        energy_cost = W_ENERGY * battery_factor * speed_sq_n

        # NO per-step cost. The checkpoint time penalty handles time pressure.

        return progress_reward - risk_cost - energy_cost

    def _check_checkpoint(self) -> float:
        if self.next_checkpoint_idx >= len(self.checkpoints):
            return 0.0

        cp = self.checkpoints[self.next_checkpoint_idx]
        cp_world = self._latlon_to_world(*cp.latlon_alt)
        dist = float(np.linalg.norm(self.pos - cp_world))

        if dist >= CHECKPOINT_RADIUS_M:
            return 0.0

        deviation = self.sim_time - cp.target_time_s
        self.checkpoint_deviations.append(deviation)
        self.next_checkpoint_idx += 1

        excess = max(0.0, abs(deviation) - TIME_TOLERANCE_S)
        if excess == 0.0:
            return 10.0   # bonus for hitting checkpoint on schedule

        penalty = W_TIME_PENALTY * self.urgency * (excess ** 2) * 0.05
        penalty = min(penalty, TIME_PENALTY_CAP)
        return -penalty + 5.0   # +5 for reaching the checkpoint at all

    def _check_termination(self) -> tuple[bool, float]:
        if np.linalg.norm(self.pos - self._goal_world) < GOAL_RADIUS_M:
            total_dev = sum(abs(d) for d in self.checkpoint_deviations)
            accuracy_bonus = 1500.0 / (1.0 + total_dev / 30.0)
            battery_bonus = self.battery * 300.0
            return True, REWARD_GOAL + accuracy_bonus + battery_bonus

        ix, iy, iz = self._world_to_voxel(self.pos)
        if self.voxel_grid.grid[ix, iy, iz] == 1:
            return True, REWARD_CRASH

        if self.pos[2] < MIN_ALT_M:
            return True, REWARD_LOW_ALT
        if self.pos[2] > MAX_ALT_M:
            return True, REWARD_HIGH_ALT

        if (self.pos[0] < 0 or self.pos[0] > self.voxel_grid.width_m or
                self.pos[1] < 0 or self.pos[1] > self.voxel_grid.length_m):
            return True, REWARD_OUT_OF_BOUNDS

        if self.battery <= 0.0:
            return True, REWARD_BATTERY_DEAD

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
            rel_goal,
            vel_n,
            wind_here,
            wind_ahead,
            [self.battery],
            [dist_cp_n],
            [np.clip(time_dev, -1.0, 1.0)],
            unit_to_cp,
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
        pass


# ======================================================================
# Reward sanity check
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("REWARD SANITY CHECK")
    print("=" * 60)
    print("Checking that 'reaching the goal' is unambiguously better")
    print("than any form of early termination.\n")

    env = AORVAEnv()
    env.reset(seed=0)

    # Estimate worst-case bleed during a flailing episode
    max_episode_steps = int(env.max_episode_time / DT)
    worst_bleed = max_episode_steps * (W_RISK * 1.0 + W_ENERGY * 3.0)
    print(f"Worst-case per-step bleed over full episode: {worst_bleed:.1f}")
    print(f"Compare to terminal rewards:")
    print(f"  Goal:           +{REWARD_GOAL:.0f} to +{REWARD_GOAL + 1500 + 300:.0f}")
    print(f"  Timeout:        {REWARD_TIMEOUT:.0f}")
    print(f"  Out of bounds:  {REWARD_OUT_OF_BOUNDS:.0f}")
    print(f"  Battery dead:   {REWARD_BATTERY_DEAD:.0f}")
    print(f"  Altitude high:  {REWARD_HIGH_ALT:.0f}")
    print(f"  Altitude low:   {REWARD_LOW_ALT:.0f}")
    print(f"  CRASH:          {REWARD_CRASH:.0f}  <-- must be the worst")
    print()

    if REWARD_CRASH < min(REWARD_TIMEOUT, REWARD_OUT_OF_BOUNDS,
                          REWARD_BATTERY_DEAD, REWARD_HIGH_ALT, REWARD_LOW_ALT):
        print("PASS  Crashing is the strictly worst terminal outcome.")
    else:
        print("FAIL  Some other termination is worse than crashing -- "
              "the agent will prefer crashing over that.")

    if abs(REWARD_TIMEOUT) < abs(REWARD_CRASH) / 10:
        print("PASS  Timeout penalty is much smaller than crash penalty.")
        print("      Agent should prefer timing out over crashing.")
    else:
        print("WARN  Timeout penalty may still incentivise early termination.")

    print("\n" + "=" * 60)
    print("NAIVE FLIGHT TEST (head straight at goal)")
    print("=" * 60)

    obs, info = env.reset(seed=0)
    total_reward = 0.0
    for t in range(20_000):
        direction = env._goal_world - env.pos
        norm = np.linalg.norm(direction)
        action = direction / max(norm, 1e-6)
        obs, r, terminated, truncated, info = env.step(action)
        total_reward += r
        if terminated or truncated:
            print(f"\nEpisode ended at step {t}: total reward {total_reward:.1f}")
            print(f"  sim_time = {info['sim_time']:.1f} s")
            print(f"  checkpoints passed: {info['checkpoints_passed']} / "
                  f"{len(env.checkpoints) - 1}")
            print(f"  battery: {info['battery']:.2%}")
            print(f"  terminated={terminated}, truncated={truncated}")
            if total_reward > 0:
                print("\n  GOOD: A naive 'head at goal' policy gets positive reward.")
                print("  RL agents will easily find this baseline and improve.")
            else:
                print("\n  CONCERN: Even a naive policy gets negative reward.")
                print("  Reward weights still need adjustment.")
            break
