"""
evaluate_agents.py  (training module - filename preserved for backwards compatibility)

Trains PPO and SAC agents on AORVAEnv using Stable Baselines 3.
This file is the training backend; use scripts/04_train_agents.py to invoke it,
or run it directly with the same CLI.

Usage
-----
    python evaluate_agents.py ppo                 # PPO only
    python evaluate_agents.py sac                 # SAC only
    python evaluate_agents.py both                # both (default)
    python evaluate_agents.py both --ppo-steps 500000 --sac-steps 300000

Outputs
-------
    models/ppo_aorva_final.zip       final PPO policy
    models/sac_aorva_final.zip       final SAC policy
    models/ppo_best/best_model.zip   best PPO checkpoint
    models/sac_best/best_model.zip   best SAC checkpoint
    logs/ppo/                        TensorBoard logs (PPO)
    logs/sac/                        TensorBoard logs (SAC)
    logs/monitor_*.csv               per-episode reward / length

Monitor training live:
    tensorboard --logdir logs

Training times (CPU only, rough estimates)
------------------------------------------
    PPO, 1M steps, 4 parallel envs : 2-4 hours
    SAC, 500k steps, single env    : 3-6 hours
    GPU (CUDA) cuts both by ~3x.
    For a quick smoke-test: pass --ppo-steps 100000 --sac-steps 50000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from aorva_env import AORVAEnv


MODEL_DIR = Path('models')
LOG_DIR = Path('logs')
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


def _best_wind_csv() -> str:
    """Return the best available wind CSV path."""
    for p in ['data/wind_spatial_real.csv', 'data/wind_historical_real.csv']:
        if Path(p).exists():
            return p
    raise FileNotFoundError(
        "No wind CSV found. Run  python scripts/00_download_real_wind.py  first."
    )


def make_env(rank: int = 0, seed: int = 0):
    """Factory that returns a thunk creating a monitored AORVAEnv."""
    wind_path = _best_wind_csv()

    def _init():
        env = AORVAEnv(wind_df_path=wind_path)
        env = Monitor(env, str(LOG_DIR / f'monitor_{rank}.csv'))
        env.reset(seed=seed + rank)
        return env
    return _init


# ----------------------------------------------------------------------
# PPO
# ----------------------------------------------------------------------
def train_ppo(total_timesteps: int = 1_000_000, n_envs: int = 4) -> None:
    print("\n" + "=" * 60)
    print(f"TRAINING PPO  ({total_timesteps:,} steps, {n_envs} envs)")
    print("=" * 60)

    if n_envs > 1:
        env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(0)])
    eval_env = DummyVecEnv([make_env(999)])

    # Hyperparameters: SB3 defaults for PPO, tuned mildly for continuous control
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log=str(LOG_DIR / 'ppo'),
    )

    callbacks = [
        CheckpointCallback(
            save_freq=max(50_000 // n_envs, 1),
            save_path=str(MODEL_DIR),
            name_prefix='ppo_aorva',
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(MODEL_DIR / 'ppo_best'),
            log_path=str(LOG_DIR / 'ppo_eval'),
            eval_freq=max(10_000 // n_envs, 1),
            n_eval_episodes=5,
            deterministic=True,
            render=False,
        ),
    ]

    model.learn(total_timesteps=total_timesteps, callback=callbacks,
                progress_bar=True)
    model.save(str(MODEL_DIR / 'ppo_aorva_final'))
    env.close()
    eval_env.close()
    print(f"\nPPO saved to {MODEL_DIR / 'ppo_aorva_final.zip'}")


# ----------------------------------------------------------------------
# SAC
# ----------------------------------------------------------------------
def train_sac(total_timesteps: int = 500_000) -> None:
    print("\n" + "=" * 60)
    print(f"TRAINING SAC  ({total_timesteps:,} steps)")
    print("=" * 60)

    # SAC is off-policy -> replay buffer. Single env is standard.
    env = DummyVecEnv([make_env(0)])
    eval_env = DummyVecEnv([make_env(999)])

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=200_000,
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
        CheckpointCallback(
            save_freq=50_000,
            save_path=str(MODEL_DIR),
            name_prefix='sac_aorva',
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(MODEL_DIR / 'sac_best'),
            log_path=str(LOG_DIR / 'sac_eval'),
            eval_freq=10_000,
            n_eval_episodes=5,
            deterministic=True,
            render=False,
        ),
    ]

    model.learn(total_timesteps=total_timesteps, callback=callbacks,
                progress_bar=True)
    model.save(str(MODEL_DIR / 'sac_aorva_final'))
    env.close()
    eval_env.close()
    print(f"\nSAC saved to {MODEL_DIR / 'sac_aorva_final.zip'}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('algorithm', nargs='?', default='both',
                        choices=['ppo', 'sac', 'both'])
    parser.add_argument('--ppo-steps', type=int, default=1_000_000)
    parser.add_argument('--sac-steps', type=int, default=500_000)
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
