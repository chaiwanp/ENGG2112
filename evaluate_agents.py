"""
train_agents.py

Train PPO and SAC agents on the AORVAEnv using Stable Baselines 3.

Usage
-----
    python train_agents.py ppo                 # PPO only
    python train_agents.py sac                 # SAC only
    python train_agents.py both                # both (default)
    python train_agents.py both --ppo-steps 500000 --sac-steps 300000

Outputs
-------
    models/ppo_aorva_final.zip       final PPO policy
    models/sac_aorva_final.zip       final SAC policy
    models/ppo_best/best_model.zip   best PPO during eval
    models/sac_best/best_model.zip   best SAC during eval
    logs/ppo/                        PPO TensorBoard logs
    logs/sac/                        SAC TensorBoard logs

Monitor training with:
    tensorboard --logdir logs

Notes on training times (rough, CPU-only)
-----------------------------------------
    PPO, 1M steps, 4 parallel envs : ~2-4 hours
    SAC, 500k steps, single env    : ~3-6 hours
GPU cuts both by ~3x. For initial debugging, drop timesteps to 100k.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv, SubprocVecEnv, VecNormalize, sync_envs_normalization,
)

from aorva_env import AORVAEnv


_BASE_DIR = Path(__file__).parent
MODEL_DIR = _BASE_DIR / 'models'
LOG_DIR = _BASE_DIR / 'logs'
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


class EpisodeLogCallback(BaseCallback):
    """
    Prints episode count + mean reward every `log_every` completed episodes.
    Writes episode rewards to a per-algo CSV so evaluation can plot
    reward-per-episode curves (distinct from the Monitor per-env files).
    """

    def __init__(self, log_every: int = 10, algo: str = 'ppo'):
        super().__init__(verbose=0)
        self.log_every = log_every
        self.algo = algo
        self._n_episodes = 0
        self._recent_rewards: list[float] = []
        self._all_rewards: list[tuple[int, int, float]] = []  # (episode, step, reward)

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self._n_episodes += 1
                ep_r = float(info['episode']['r'])
                self._recent_rewards.append(ep_r)
                self._all_rewards.append(
                    (self._n_episodes, self.num_timesteps, ep_r)
                )
                if self._n_episodes % self.log_every == 0:
                    mean_r = np.mean(self._recent_rewards[-self.log_every:])
                    print(
                        f"  [{self.algo.upper()} ep {self._n_episodes:6d}] "
                        f"steps={self.num_timesteps:9,}  "
                        f"mean_reward(last {self.log_every})={mean_r:8.1f}"
                    )
        return True

    def _on_training_end(self) -> None:
        if not self._all_rewards:
            return
        import pandas as pd
        df = pd.DataFrame(self._all_rewards,
                          columns=['episode', 'total_steps', 'reward'])
        out = LOG_DIR / f'{self.algo}_episode_log.csv'
        df.to_csv(out, index=False)
        print(f"  Episode log saved to {out}  ({self._n_episodes} episodes)")


def make_env(rank: int = 0, seed: int = 0, algo: str = 'ppo'):
    """Factory that returns a thunk creating a monitored AORVAEnv."""
    def _init():
        env = AORVAEnv()
        env = Monitor(env, str(LOG_DIR / f'{algo}_monitor_{rank}.csv'))
        env.reset(seed=seed + rank)
        return env
    return _init


class SyncedEvalCallback(EvalCallback):
    """
    EvalCallback that syncs VecNormalize running statistics from the training
    env to the eval env before each evaluation, so the eval env uses the same
    observation normalisation as the policy being tested.
    """

    def __init__(self, eval_env, train_env: VecNormalize, **kwargs):
        super().__init__(eval_env, **kwargs)
        self._train_env = train_env

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            sync_envs_normalization(self._train_env, self.eval_env)
        return super()._on_step()


# ----------------------------------------------------------------------
# PPO
# ----------------------------------------------------------------------
def train_ppo(total_timesteps: int = 1_000_000, n_envs: int = 4) -> None:
    print("\n" + "=" * 60)
    print(f"TRAINING PPO  ({total_timesteps:,} steps, {n_envs} envs)")
    print("=" * 60)

    if n_envs > 1:
        env = SubprocVecEnv([make_env(i, algo='ppo') for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(0, algo='ppo')])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    eval_env = DummyVecEnv([make_env(999, algo='ppo')])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    eval_freq = max(50_000 // n_envs, 1)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        # Linear decay: large updates early when policy is far from optimal,
        # small updates late to avoid destabilising a converged policy.
        learning_rate=lambda progress: 3e-4 * progress,
        n_steps=1024,        # halved from 2048 → 2× more frequent gradient updates
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log=str(LOG_DIR / 'ppo'),
    )

    callbacks = [
        EpisodeLogCallback(log_every=10, algo='ppo'),
        CheckpointCallback(
            save_freq=max(50_000 // n_envs, 1),
            save_path=str(MODEL_DIR),
            name_prefix='ppo_aorva',
        ),
        SyncedEvalCallback(
            eval_env,
            train_env=env,
            best_model_save_path=str(MODEL_DIR / 'ppo_best'),
            log_path=str(LOG_DIR / 'ppo_eval'),
            eval_freq=eval_freq,
            n_eval_episodes=10,
            deterministic=True,
            render=False,
        ),
    ]

    model.learn(total_timesteps=total_timesteps, callback=callbacks,
                progress_bar=True)
    model.save(str(MODEL_DIR / 'ppo_aorva_final'))
    # VecNormalize stats must be saved alongside the model; without them the
    # loaded policy would receive un-normalised observations and perform poorly.
    env.save(str(MODEL_DIR / 'ppo_vecnorm.pkl'))
    env.close()
    eval_env.close()
    print(f"\nPPO saved to {MODEL_DIR / 'ppo_aorva_final.zip'}")
    print(f"PPO VecNormalize stats saved to {MODEL_DIR / 'ppo_vecnorm.pkl'}")


# ----------------------------------------------------------------------
# SAC
# ----------------------------------------------------------------------
def train_sac(total_timesteps: int = 1_000_000) -> None:
    print("\n" + "=" * 60)
    print(f"TRAINING SAC  ({total_timesteps:,} steps)")
    print("=" * 60)

    # SAC is off-policy -> replay buffer. Single env is standard.
    env = DummyVecEnv([make_env(0, algo='sac')])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    eval_env = DummyVecEnv([make_env(999, algo='sac')])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=500_000,   # increased from 200k — keeps ~50% of 1M-step experience
        learning_starts=10_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef='auto',
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log=str(LOG_DIR / 'sac'),
    )

    callbacks = [
        EpisodeLogCallback(log_every=10, algo='sac'),
        CheckpointCallback(
            save_freq=50_000,
            save_path=str(MODEL_DIR),
            name_prefix='sac_aorva',
        ),
        SyncedEvalCallback(
            eval_env,
            train_env=env,
            best_model_save_path=str(MODEL_DIR / 'sac_best'),
            log_path=str(LOG_DIR / 'sac_eval'),
            eval_freq=50_000,
            n_eval_episodes=10,
            deterministic=True,
            render=False,
        ),
    ]

    model.learn(total_timesteps=total_timesteps, callback=callbacks,
                progress_bar=True)
    model.save(str(MODEL_DIR / 'sac_aorva_final'))
    env.save(str(MODEL_DIR / 'sac_vecnorm.pkl'))
    env.close()
    eval_env.close()
    print(f"\nSAC saved to {MODEL_DIR / 'sac_aorva_final.zip'}")
    print(f"SAC VecNormalize stats saved to {MODEL_DIR / 'sac_vecnorm.pkl'}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('algorithm', nargs='?', default='both',
                        choices=['ppo', 'sac', 'both'])
    parser.add_argument('--ppo-steps', type=int, default=1_000_000)
    parser.add_argument('--sac-steps', type=int, default=1_000_000)
    parser.add_argument('--n-envs', type=int, default=4,
                        help='Parallel envs for PPO')
    args = parser.parse_args()

    if args.algorithm in ('ppo', 'both'):
        train_ppo(args.ppo_steps, args.n_envs)
    if args.algorithm in ('sac', 'both'):
        train_sac(args.sac_steps)

    print("\nAll training complete.")


if __name__ == '__main__':
    main()
