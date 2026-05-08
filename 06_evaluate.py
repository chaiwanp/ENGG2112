"""
Step 6 - Evaluate trained PPO and SAC agents (Phase 4 of AORVA pipeline).

Loads trained models, runs 100 test episodes each with randomised wind
conditions, and produces all Phase 4 metrics and visualisations:

Metrics (Phase 4.2)
-------------------
  RMSE of arrival-time deviation at each of the 15 checkpoints
  Safety violation rate (altitude / no-fly zone / collision)
  Success rate (drone reached Liverpool Hospital)
  Energy efficiency (mean remaining battery at goal)
  delta_sigma = sigma_ambulance - sigma_drone  (headline reliability result)

Outputs
-------
  outputs/eval_training_curves.png     reward vs timestep (TensorBoard CSV)
  outputs/eval_checkpoint_timing.png   actual vs target arrival times
  outputs/eval_trajectory_map.png      RL paths overlaid on corridor map
  outputs/eval_comparison_table.txt    PPO vs SAC vs Ambulance summary

Ambulance baseline (Phase 4.3)
-------------------------------
The Westmead->Liverpool driving-time baseline is estimated from historical
Google Maps traffic data for this corridor.  Mean ~ 35 min, sigma ~ 12 min
(roughly 2x variation between off-peak and peak congestion).
Replace AMBULANCE_MEAN_S and AMBULANCE_STD_S with real measurements when
available (see Phase 4.3 in the pipeline walkthrough).

Prerequisites: run steps 00-05 and at least one training run (step 04).

Usage:
    python scripts/06_evaluate.py                    # evaluate all found models
    python scripts/06_evaluate.py --episodes 50      # faster smoke test
    python scripts/06_evaluate.py --ppo-only
    python scripts/06_evaluate.py --sac-only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aorva_env import AORVAEnv

# -- Ambulance baseline (Phase 4.3) ------------------------------------------
# Source: Westmead->Liverpool ~30 km by road.
# Off-peak ~25 min, peak ~65+ min. Distribution parameters from TomTom/Waze.
AMBULANCE_MEAN_S = 35.0 * 60    # 35 minutes in seconds
AMBULANCE_STD_S  = 12.0 * 60    # 12 minutes std dev


# -- Model search paths -------------------------------------------------------
MODEL_CANDIDATES = {
    'PPO': [
        'models/ppo_best/best_model.zip',
        'models/ppo_aorva_final.zip',
    ],
    'SAC': [
        'models/sac_best/best_model.zip',
        'models/sac_aorva_final.zip',
    ],
}


def _find_model(algo: str) -> Path | None:
    for p in MODEL_CANDIDATES[algo]:
        if Path(p).exists():
            return Path(p)
    return None


def _load_model(algo: str, model_path: Path, env: AORVAEnv):
    """Load a trained SB3 model."""
    from stable_baselines3 import PPO, SAC
    cls = PPO if algo == 'PPO' else SAC
    return cls.load(str(model_path), env=env)


def run_episodes(model, env: AORVAEnv, n_episodes: int, deterministic: bool = True
                 ) -> list[dict]:
    """
    Run `n_episodes` and return a list of episode result dicts.

    Each dict has:
        success          bool
        total_time_s     float
        battery_left     float
        checkpoint_devs  list[float]   (actual - target for each checkpoint)
        violated         bool          (any safety violation)
        trajectory       list[np.ndarray]  3-D positions
    """
    results = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        done = False
        ep_violated = False

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Check for safety events via termination reason (reward signals)
            if terminated and reward < -200:
                ep_violated = True

        results.append({
            'success':         terminated and reward > 0,
            'total_time_s':    info['sim_time'],
            'battery_left':    info['battery'],
            'checkpoint_devs': info['checkpoint_deviations'],
            'violated':        ep_violated,
            'trajectory':      info.get('trajectory', []),
        })

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"    Episode {ep + 1}/{n_episodes}: "
                  f"success={results[-1]['success']}  "
                  f"time={results[-1]['total_time_s']:.0f}s  "
                  f"battery={results[-1]['battery_left']:.2f}")

    return results


def compute_metrics(results: list[dict], n_checkpoints: int = 15) -> dict:
    """Aggregate episode results into scalar metrics."""
    successes = [r['success'] for r in results]
    times     = [r['total_time_s'] for r in results if r['success']]
    batteries = [r['battery_left'] for r in results if r['success']]
    violated  = [r['violated'] for r in results]

    # Checkpoint deviation per checkpoint index
    all_devs  = [r['checkpoint_devs'] for r in results if r['checkpoint_devs']]
    flat_devs = [d for ep in all_devs for d in ep]

    # Per-checkpoint mean absolute deviation
    cp_mae = []
    for k in range(n_checkpoints - 1):
        devs_k = [ep[k] for ep in all_devs if k < len(ep)]
        cp_mae.append(float(np.mean(np.abs(devs_k))) if devs_k else float('nan'))

    rmse = float(np.sqrt(np.mean(np.square(flat_devs)))) if flat_devs else float('nan')

    return {
        'n_episodes':       len(results),
        'success_rate':     float(np.mean(successes)),
        'violation_rate':   float(np.mean(violated)),
        'mean_time_s':      float(np.mean(times))   if times else float('nan'),
        'std_time_s':       float(np.std(times))    if times else float('nan'),
        'mean_battery':     float(np.mean(batteries)) if batteries else float('nan'),
        'rmse_dev_s':       rmse,
        'cp_mae_s':         cp_mae,
    }


# -- Visualisation helpers ----------------------------------------------------

def plot_checkpoint_timing(metrics_dict: dict, env: AORVAEnv,
                            out_path: str) -> None:
    """Actual vs target arrival time at each checkpoint."""
    fig, axes = plt.subplots(1, len(metrics_dict), figsize=(7 * len(metrics_dict), 6),
                              squeeze=False)
    target_times = [c.target_time_s for c in env.checkpoints[1:]]

    for col, (algo, m) in enumerate(metrics_dict.items()):
        ax = axes[0][col]
        cp_mae = m['cp_mae_s']
        n = min(len(cp_mae), len(target_times))
        actual = [target_times[k] + cp_mae[k] for k in range(n)]

        ax.plot(target_times[:n], target_times[:n], '--',
                color='grey', linewidth=1.2, label='Perfect (y=x)')
        ax.scatter(target_times[:n], actual,
                   c=np.abs(cp_mae[:n]), cmap='RdYlGn_r', s=80, zorder=5,
                   edgecolors='white', linewidths=0.6, label='Mean actual')
        for k in range(n):
            ax.annotate(str(k + 1),
                        (target_times[k], actual[k]),
                        fontsize=7, ha='left', va='bottom', color='navy')

        ax.set_xlabel('Target arrival time (s)', fontsize=9)
        ax.set_ylabel('Mean actual arrival time (s)', fontsize=9)
        ax.set_title(
            f'{algo} - Checkpoint timing\n'
            f'RMSE = {m["rmse_dev_s"]:.1f} s  |  '
            f'success = {m["success_rate"]*100:.0f}%',
            fontsize=10,
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Phase 4 - Checkpoint Timing: Actual vs Target',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def plot_trajectory_map(results_dict: dict, env: AORVAEnv,
                         out_path: str, max_trajs: int = 20) -> None:
    """Sample RL trajectories overlaid on the voxel grid."""
    vg  = env.voxel_grid
    vs  = vg.voxel_size_m
    n   = len(results_dict)

    fig, axes = plt.subplots(1, n, figsize=(11 * n, 9), squeeze=False)

    for col, (algo, results) in enumerate(results_dict.items()):
        ax = axes[0][col]

        # Building projection
        occ = np.max(vg.grid, axis=2).T
        ax.imshow(occ, origin='lower', cmap='Greys', alpha=0.3,
                  extent=[0, vg.nx, 0, vg.ny], aspect='auto')

        # A* reference path (from env checkpoints)
        cp_xy = np.array([
            [c.voxel[0], c.voxel[1]] for c in env.checkpoints
        ])
        ax.plot(cp_xy[:, 0], cp_xy[:, 1], 'b--', linewidth=1.8,
                alpha=0.6, label='A* reference', zorder=4)

        # RL trajectories
        trajs = [r['trajectory'] for r in results if r['trajectory']]
        sampled = trajs[:max_trajs]
        for traj in sampled:
            pts = np.array(traj)
            gx  = pts[:, 0] / vs
            gy  = pts[:, 1] / vs
            col_val = 'limegreen' if True else 'tomato'
            ax.plot(gx, gy, '-', color='limegreen', linewidth=0.8,
                    alpha=0.4, zorder=3)

        # Hospital markers
        wx, wy, _ = vg.latlon_to_grid(-33.8078, 150.9875, 0)
        lx, ly, _ = vg.latlon_to_grid(-33.9173, 150.9233, 0)
        ax.plot(wx, wy, 'b*', markersize=20, markeredgecolor='white',
                markeredgewidth=1.5, label='Westmead', zorder=8)
        ax.plot(lx, ly, 'r*', markersize=20, markeredgecolor='white',
                markeredgewidth=1.5, label='Liverpool', zorder=8)

        m = compute_metrics(results)
        ax.set_title(
            f'{algo} - Sample Trajectories (n={len(sampled)})\n'
            f'Success {m["success_rate"]*100:.0f}%  |  '
            f'Violation {m["violation_rate"]*100:.0f}%',
            fontsize=10,
        )
        ax.set_xlabel('Grid X  (west -> east)', fontsize=9)
        ax.set_ylabel('Grid Y  (south -> north)', fontsize=9)
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
        ax.set_xlim(0, vg.nx)
        ax.set_ylim(0, vg.ny)

    plt.suptitle('Phase 4 - Trained Agent Trajectories vs A* Reference',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def plot_training_curves(out_path: str) -> bool:
    """Plot reward curves from TensorBoard monitor CSV files if available."""
    monitor_files = list(Path('logs').glob('monitor_*.csv'))
    if not monitor_files:
        print("No monitor CSV files found - skipping training curves plot.")
        return False

    fig, ax = plt.subplots(figsize=(12, 6))
    for mf in sorted(monitor_files)[:8]:
        try:
            df = pd.read_csv(mf, skiprows=1)
            if 'r' in df.columns:
                smoothed = df['r'].rolling(window=20, min_periods=1).mean()
                ax.plot(smoothed.values, linewidth=1.2, alpha=0.7,
                        label=mf.stem)
        except Exception:
            continue

    ax.set_xlabel('Episode', fontsize=9)
    ax.set_ylabel('Episode reward (smoothed)', fontsize=9)
    ax.set_title('Training Curves - Episode Reward vs Episode\n'
                 '(rolling mean, window=20)', fontsize=11)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")
    return True


def print_comparison_table(metrics_dict: dict, out_path: str) -> None:
    """Print and save the Phase 4 final comparison table."""
    header = f"{'Metric':<30} {'PPO':>12} {'SAC':>12} {'Ambulance':>12}"
    sep    = "-" * len(header)

    def fmt(v, fmt_str='.1f'):
        return 'N/A' if (v is None or (isinstance(v, float) and np.isnan(v))) \
               else format(v, fmt_str)

    lines = [
        "=" * len(header),
        "AORVA Phase 4 - Final Comparison Table",
        "=" * len(header),
        header,
        sep,
    ]

    def row(label, ppo_v, sac_v, amb_v, fmt_str='.2f'):
        return (f"{label:<30} "
                f"{fmt(ppo_v, fmt_str):>12} "
                f"{fmt(sac_v, fmt_str):>12} "
                f"{fmt(amb_v, fmt_str):>12}")

    m_ppo = metrics_dict.get('PPO', {})
    m_sac = metrics_dict.get('SAC', {})

    lines += [
        row("Success rate (%)",
            m_ppo.get('success_rate', float('nan')) * 100,
            m_sac.get('success_rate', float('nan')) * 100,
            None, '.1f'),
        row("Safety violation rate (%)",
            m_ppo.get('violation_rate', float('nan')) * 100,
            m_sac.get('violation_rate', float('nan')) * 100,
            None, '.1f'),
        row("Mean delivery time (min)",
            m_ppo.get('mean_time_s', float('nan')) / 60,
            m_sac.get('mean_time_s', float('nan')) / 60,
            AMBULANCE_MEAN_S / 60, '.1f'),
        row("Std dev time sigma (min)",
            m_ppo.get('std_time_s', float('nan')) / 60,
            m_sac.get('std_time_s', float('nan')) / 60,
            AMBULANCE_STD_S / 60, '.1f'),
        row("RMSE checkpoint dev (s)",
            m_ppo.get('rmse_dev_s', float('nan')),
            m_sac.get('rmse_dev_s', float('nan')),
            None, '.1f'),
        row("Mean battery remaining (%)",
            (m_ppo.get('mean_battery', float('nan')) or 0) * 100,
            (m_sac.get('mean_battery', float('nan')) or 0) * 100,
            None, '.1f'),
        sep,
    ]

    # delta_sigma = sigma_ambulance - sigma_drone
    sigma_amb = AMBULANCE_STD_S / 60
    for algo, m in [('PPO', m_ppo), ('SAC', m_sac)]:
        sigma_d = m.get('std_time_s', float('nan')) / 60
        delta_s = sigma_amb - sigma_d if not np.isnan(sigma_d) else float('nan')
        sign    = '>' if (not np.isnan(delta_s) and delta_s > 0) else '<'
        verdict = ('drone MORE reliable than ambulance' if
                   (not np.isnan(delta_s) and delta_s > 0) else
                   'drone LESS reliable than ambulance')
        lines.append(
            f"delta_sigma ({algo}): sigma_amb ({sigma_amb:.1f} min) - "
            f"sigma_drone ({sigma_d:.1f} min) = {fmt(delta_s, '.1f')} min  "
            f"-> {verdict}"
        )

    lines += [
        "=" * len(header),
        "",
        "Ambulance baseline: Westmead->Liverpool, mean ~ 35 min, sigma ~ 12 min",
        "(estimated from TomTom traffic data; replace with measured data for publication)",
    ]

    table_str = "\n".join(lines)
    print("\n" + table_str)

    with open(out_path, 'w') as f:
        f.write(table_str + "\n")
    print(f"\nSaved {out_path}")


# -- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AORVA Step 6 - Evaluate trained RL agents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--episodes', type=int, default=100,
                        help='Test episodes per agent')
    parser.add_argument('--ppo-only', action='store_true')
    parser.add_argument('--sac-only', action='store_true')
    parser.add_argument('--stochastic', action='store_true',
                        help='Use stochastic policy (default: deterministic)')
    args = parser.parse_args()

    os.makedirs('outputs', exist_ok=True)
    deterministic = not args.stochastic

    algos_to_eval = []
    if not args.sac_only:
        algos_to_eval.append('PPO')
    if not args.ppo_only:
        algos_to_eval.append('SAC')

    print("=" * 60)
    print(f"AORVA Phase 4 - Agent Evaluation  ({args.episodes} episodes each)")
    print("=" * 60)

    # Build shared environment
    env = AORVAEnv()

    results_by_algo = {}
    metrics_by_algo = {}

    for algo in algos_to_eval:
        model_path = _find_model(algo)
        if model_path is None:
            print(f"\n{algo}: no trained model found in models/. "
                  f"Run scripts/04_train_agents.py {algo.lower()} first.")
            continue

        print(f"\n-- {algo} ---------------------------------")
        print(f"   Model: {model_path}")
        model = _load_model(algo, model_path, env)

        print(f"   Running {args.episodes} episodes "
              f"({'deterministic' if deterministic else 'stochastic'})...")
        results = run_episodes(model, env, args.episodes, deterministic)
        results_by_algo[algo] = results

        m = compute_metrics(results, n_checkpoints=len(env.checkpoints))
        metrics_by_algo[algo] = m
        print(f"\n   {algo} summary:")
        print(f"     Success rate:        {m['success_rate']*100:.1f}%")
        print(f"     Violation rate:      {m['violation_rate']*100:.1f}%")
        print(f"     Mean time:           {m['mean_time_s']/60:.1f} min")
        print(f"     Std dev time:        {m['std_time_s']/60:.1f} min")
        print(f"     RMSE checkpoint dev: {m['rmse_dev_s']:.1f} s")
        print(f"     Mean battery left:   {(m['mean_battery'] or 0)*100:.1f}%")

    if not results_by_algo:
        print("\nNo models evaluated. Train at least one agent first.")
        sys.exit(1)

    # -- Generate all Phase 4 outputs --------------------------------------
    print("\n-- Generating Phase 4 visualisations --")

    plot_training_curves('outputs/eval_training_curves.png')

    if metrics_by_algo:
        plot_checkpoint_timing(metrics_by_algo, env,
                                'outputs/eval_checkpoint_timing.png')

    if results_by_algo:
        plot_trajectory_map(results_by_algo, env,
                             'outputs/eval_trajectory_map.png')

    print_comparison_table(metrics_by_algo,
                            'outputs/eval_comparison_table.txt')

    print("\nStep 6 complete.")


if __name__ == "__main__":
    main()
