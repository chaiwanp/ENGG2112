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
  outputs/eval_training_curves.png       reward vs episode (from training logs)
  outputs/eval_checkpoint_timing.png     actual vs target arrival times
  outputs/eval_trajectory_map.png        RL paths overlaid on corridor map
  outputs/eval_reward_distribution.png   violin plot of episode rewards
  outputs/eval_termination_breakdown.png why each episode ended
  outputs/eval_delivery_time.png         drone vs ambulance delivery time violin
  outputs/eval_comparison_table.txt      PPO vs SAC vs Ambulance summary

Ambulance baseline (Phase 4.3)
-------------------------------
Westmead->Liverpool driving-time baseline estimated from TomTom/Waze.
Mean ~ 35 min, sigma ~ 12 min.  Replace when measured data is available.

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
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aorva_env import AORVAEnv

# -- Ambulance baseline (Phase 4.3) ------------------------------------------
AMBULANCE_MEAN_S = 35.0 * 60    # 35 minutes in seconds
AMBULANCE_STD_S  = 12.0 * 60    # 12 minutes std dev

# -- Colour scheme -----------------------------------------------------------
ALGO_COLORS = {'PPO': 'steelblue', 'SAC': 'tomato'}

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
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    cls = PPO if algo == 'PPO' else SAC

    # Look for VecNormalize stats saved alongside the model during training.
    vecnorm_candidates = [
        model_path.parent / f'{algo.lower()}_vecnorm.pkl',
        Path('models') / f'{algo.lower()}_vecnorm.pkl',
    ]
    vecnorm_path = next((p for p in vecnorm_candidates if p.exists()), None)

    if vecnorm_path:
        # Wrap with the same normalisation the policy was trained under.
        dummy = DummyVecEnv([lambda: env])
        vec_env = VecNormalize.load(str(vecnorm_path), dummy)
        vec_env.training = False
        vec_env.norm_reward = False
        model = cls.load(str(model_path), env=vec_env)
        print(f"   Loaded VecNormalize stats from {vecnorm_path}")
        return model, vec_env
    else:
        model = cls.load(str(model_path), env=env)
        return model, None


# ============================================================================
# Episode runner
# ============================================================================

def run_episodes(model, env: AORVAEnv, n_episodes: int,
                 vec_norm=None, deterministic: bool = True) -> list[dict]:
    """
    Run `n_episodes` and return a list of episode result dicts.

    Each dict has:
        success              bool
        total_time_s         float
        battery_left         float
        checkpoint_devs      list[float]
        violated             bool
        trajectory           list[np.ndarray]
        total_reward         float
        termination_reason   str

    Parameters
    ----------
    vec_norm : VecNormalize | None
        If the model was trained with VecNormalize, pass the loaded wrapper here
        so observations are normalised before being fed to the policy.
    """
    results = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        if vec_norm is not None:
            obs = vec_norm.normalize_obs(obs.reshape(1, -1)).flatten()

        done = False
        ep_violated = False
        total_reward = 0.0
        terminated = False

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            if vec_norm is not None:
                obs = vec_norm.normalize_obs(obs.reshape(1, -1)).flatten()
            total_reward += reward
            done = terminated or truncated

            if terminated and reward < -200:
                ep_violated = True

        # Use the termination reason flag directly — more reliable than checking
        # the final step's reward, which includes per-step shaping terms.
        term_reason = getattr(env, '_termination_reason', None) or 'TIMEOUT'
        success = term_reason == 'GOAL REACHED'

        results.append({
            'success':             success,
            'total_time_s':        info['sim_time'],
            'battery_left':        info['battery'],
            'checkpoint_devs':     info['checkpoint_deviations'],
            'violated':            ep_violated,
            'trajectory':          info.get('trajectory', []),
            'total_reward':        total_reward,
            'termination_reason':  term_reason,
        })

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"    Episode {ep + 1}/{n_episodes}: "
                  f"success={results[-1]['success']}  "
                  f"time={results[-1]['total_time_s']:.0f}s  "
                  f"battery={results[-1]['battery_left']:.2f}  "
                  f"reason={term_reason}")

    return results


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(results: list[dict], n_checkpoints: int = 15) -> dict:
    successes = [r['success'] for r in results]
    times     = [r['total_time_s'] for r in results if r['success']]
    batteries = [r['battery_left'] for r in results if r['success']]
    violated  = [r['violated'] for r in results]
    rewards   = [r['total_reward'] for r in results]

    all_devs  = [r['checkpoint_devs'] for r in results if r['checkpoint_devs']]
    flat_devs = [d for ep in all_devs for d in ep]

    cp_mae = []
    for k in range(n_checkpoints - 1):
        devs_k = [ep[k] for ep in all_devs if k < len(ep)]
        cp_mae.append(float(np.mean(np.abs(devs_k))) if devs_k else float('nan'))

    rmse = float(np.sqrt(np.mean(np.square(flat_devs)))) if flat_devs else float('nan')

    return {
        'n_episodes':     len(results),
        'success_rate':   float(np.mean(successes)),
        'violation_rate': float(np.mean(violated)),
        'mean_time_s':    float(np.mean(times))    if times else float('nan'),
        'std_time_s':     float(np.std(times))     if times else float('nan'),
        'mean_battery':   float(np.mean(batteries)) if batteries else float('nan'),
        'rmse_dev_s':     rmse,
        'cp_mae_s':       cp_mae,
        'mean_reward':    float(np.mean(rewards)),
        'std_reward':     float(np.std(rewards)),
        'all_rewards':    rewards,
    }


# ============================================================================
# Plot: training curves (reward per episode, markers every 10)
# ============================================================================

def plot_training_curves(out_path: str) -> bool:
    """
    Reads per-algo episode log CSVs (written by EpisodeLogCallback) and the
    Monitor CSVs as fallback.  Plots:
      top panel   : reward vs episode number, diamond marker every 10 eps
      bottom panel: reward vs cumulative environment steps
    """
    log_dir = Path('logs')
    if not log_dir.exists():
        print("No logs/ directory - skipping training curves plot.")
        return False

    fig, (ax_ep, ax_step) = plt.subplots(2, 1, figsize=(14, 10))
    found_any = False

    for algo in ('ppo', 'sac'):
        color = ALGO_COLORS[algo.upper()]

        # Prefer the per-algo episode log (one row per episode, all envs merged)
        ep_log = log_dir / f'{algo}_episode_log.csv'
        if ep_log.exists():
            df = pd.read_csv(ep_log)
            df = df.rename(columns={'reward': 'r', 'total_steps': 'cum_steps'})
        else:
            # Fall back: combine Monitor CSVs (ppo_monitor_*.csv or monitor_*.csv)
            files = sorted(log_dir.glob(f'{algo}_monitor_*.csv'))
            if not files:
                files = sorted(log_dir.glob('monitor_*.csv'))
            if not files:
                continue

            dfs = []
            for mf in files:
                try:
                    raw = pd.read_csv(mf, skiprows=1)
                    dfs.append(raw)
                except Exception:
                    continue
            if not dfs:
                continue

            df = pd.concat(dfs, ignore_index=True)
            if 't' in df.columns:
                df = df.sort_values('t').reset_index(drop=True)
            df['episode'] = np.arange(1, len(df) + 1)
            df['cum_steps'] = df['l'].cumsum() if 'l' in df.columns else df.index

        if 'r' not in df.columns or len(df) == 0:
            continue

        found_any = True
        episodes   = df['episode'].values
        raw_r      = df['r'].values
        smoothed   = pd.Series(raw_r).rolling(window=20, min_periods=1).mean().values
        cum_steps  = df['cum_steps'].values if 'cum_steps' in df.columns else episodes

        label = algo.upper()

        # -- Episode axis --
        ax_ep.plot(episodes, raw_r,   color=color, linewidth=0.4, alpha=0.2)
        ax_ep.plot(episodes, smoothed, color=color, linewidth=1.6,
                   alpha=0.9, label=f'{label} (smoothed, window=20)')
        # Diamond marker every 10 episodes
        mask = (episodes % 10 == 0)
        ax_ep.scatter(episodes[mask], smoothed[mask], color=color,
                      s=30, marker='D', zorder=5,
                      label=f'{label} (every 10 eps)')

        # -- Step axis --
        ax_step.plot(cum_steps, raw_r,    color=color, linewidth=0.4, alpha=0.2)
        ax_step.plot(cum_steps, smoothed,  color=color, linewidth=1.6,
                     alpha=0.9, label=f'{label}')

    if not found_any:
        print("No training log files found - skipping training curves plot.")
        plt.close()
        return False

    ax_ep.set_xlabel('Episode number', fontsize=10)
    ax_ep.set_ylabel('Episode reward', fontsize=10)
    ax_ep.set_title(
        'Training curves — reward per episode\n'
        '(diamond markers every 10 episodes; faint line = raw, solid = smoothed)',
        fontsize=11,
    )
    ax_ep.legend(fontsize=8, loc='lower right')
    ax_ep.grid(True, alpha=0.3)

    ax_step.set_xlabel('Cumulative environment steps', fontsize=10)
    ax_step.set_ylabel('Episode reward', fontsize=10)
    ax_step.set_title('Training curves — reward vs environment steps', fontsize=11)
    ax_step.legend(fontsize=8, loc='lower right')
    ax_step.grid(True, alpha=0.3)
    ax_step.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{x/1e3:.0f}k')
    )

    plt.suptitle('AORVA Training Progress', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")
    return True


# ============================================================================
# Plot: checkpoint timing
# ============================================================================

def plot_checkpoint_timing(metrics_dict: dict, env: AORVAEnv,
                            out_path: str) -> None:
    fig, axes = plt.subplots(1, len(metrics_dict),
                              figsize=(7 * len(metrics_dict), 6), squeeze=False)
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


# ============================================================================
# Plot: trajectory map (success = green, failure = red)
# ============================================================================

def plot_trajectory_map(results_dict: dict, env: AORVAEnv,
                         out_path: str, max_trajs: int = 20) -> None:
    vg  = env.voxel_grid
    vs  = vg.voxel_size_m
    n   = len(results_dict)

    fig, axes = plt.subplots(1, n, figsize=(11 * n, 9), squeeze=False)

    for col, (algo, results) in enumerate(results_dict.items()):
        ax = axes[0][col]

        occ = np.max(vg.grid, axis=2).T
        ax.imshow(occ, origin='lower', cmap='Greys', alpha=0.3,
                  extent=[0, vg.nx, 0, vg.ny], aspect='auto')

        cp_xy = np.array([[c.voxel[0], c.voxel[1]] for c in env.checkpoints])
        ax.plot(cp_xy[:, 0], cp_xy[:, 1], 'b--', linewidth=1.8,
                alpha=0.6, label='A* reference', zorder=4)

        # Sample trajectories, colour by success/failure
        paired = [(r['trajectory'], r['success'])
                  for r in results if r['trajectory']][:max_trajs]
        success_plotted = failure_plotted = False
        for traj, success in paired:
            pts   = np.array(traj)
            gx    = pts[:, 0] / vs
            gy    = pts[:, 1] / vs
            color = 'limegreen' if success else 'tomato'
            label = None
            if success and not success_plotted:
                label = 'Success'
                success_plotted = True
            elif not success and not failure_plotted:
                label = 'Failure'
                failure_plotted = True
            ax.plot(gx, gy, '-', color=color, linewidth=0.8,
                    alpha=0.4, zorder=3, label=label)

        wx, wy, _ = vg.latlon_to_grid(-33.8078, 150.9875, 0)
        lx, ly, _ = vg.latlon_to_grid(-33.9173, 150.9233, 0)
        ax.plot(wx, wy, 'b*', markersize=20, markeredgecolor='white',
                markeredgewidth=1.5, label='Westmead', zorder=8)
        ax.plot(lx, ly, 'r*', markersize=20, markeredgecolor='white',
                markeredgewidth=1.5, label='Liverpool', zorder=8)

        m = compute_metrics(results)
        ax.set_title(
            f'{algo} - Sample Trajectories (n={len(paired)})\n'
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


# ============================================================================
# Plot: episode reward distribution (violin)
# ============================================================================

def plot_reward_distribution(results_dict: dict, out_path: str) -> None:
    """Violin + box overlay showing total episode reward per agent."""
    algos   = list(results_dict.keys())
    rewards = [[r['total_reward'] for r in results_dict[a]] for a in algos]

    fig, ax = plt.subplots(figsize=(7, 6))

    parts = ax.violinplot(rewards, positions=range(len(algos)),
                          showmeans=False, showmedians=False, showextrema=False)
    for i, (body, algo) in enumerate(zip(parts['bodies'], algos)):
        body.set_facecolor(ALGO_COLORS[algo])
        body.set_alpha(0.55)

    # Box plot overlay
    bp = ax.boxplot(rewards, positions=range(len(algos)),
                    widths=0.15, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))
    for patch, algo in zip(bp['boxes'], algos):
        patch.set_facecolor(ALGO_COLORS[algo])
        patch.set_alpha(0.7)

    # Mean annotation
    for i, (algo, rw) in enumerate(zip(algos, rewards)):
        mu = np.mean(rw)
        ax.text(i + 0.14, mu, f' μ={mu:.0f}', va='center',
                fontsize=9, color=ALGO_COLORS[algo])

    ax.set_xticks(range(len(algos)))
    ax.set_xticklabels(algos, fontsize=12)
    ax.set_ylabel('Total episode reward', fontsize=10)
    ax.set_title('Episode Reward Distribution by Agent\n'
                 '(violin = density, box = IQR, line = median)', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


# ============================================================================
# Plot: termination reason breakdown
# ============================================================================

_REASON_COLORS = {
    'GOAL REACHED': '#2ecc71',
    'COLLISION':    '#e74c3c',
    'TOO LOW':      '#e67e22',
    'TOO HIGH':     '#9b59b6',
    'OUT OF BOUNDS':'#8B4513',
    'BATTERY DEAD': '#f1c40f',
    'TIMEOUT':      '#95a5a6',
}


def _categorise_reason(raw: str | None) -> str:
    if not raw:
        return 'TIMEOUT'
    for key in _REASON_COLORS:
        if key in raw.upper():
            return key
    return 'TIMEOUT'


def plot_termination_breakdown(results_dict: dict, out_path: str) -> None:
    """Stacked bar chart: why episodes ended, per agent."""
    algos = list(results_dict.keys())
    reason_counts: dict[str, Counter] = {}

    all_cats: set[str] = set()
    for algo, results in results_dict.items():
        counts: Counter = Counter()
        for r in results:
            counts[_categorise_reason(r.get('termination_reason'))] += 1
        reason_counts[algo] = counts
        all_cats.update(counts.keys())

    # Order: goal first, then failures
    ordered = ['GOAL REACHED'] + [c for c in _REASON_COLORS if c != 'GOAL REACHED'
                                    and c in all_cats]

    fig, ax = plt.subplots(figsize=(9, 6))
    x       = np.arange(len(algos))
    bottoms = np.zeros(len(algos))

    for cat in ordered:
        n_eps   = [len(results_dict[a]) for a in algos]
        pcts    = [reason_counts[a].get(cat, 0) / n * 100
                   for a, n in zip(algos, n_eps)]
        ax.bar(x, pcts, 0.55, bottom=bottoms,
               color=_REASON_COLORS.get(cat, 'lightblue'),
               label=cat, alpha=0.88)
        # Label segments > 5%
        for xi, (pct, bot) in enumerate(zip(pcts, bottoms)):
            if pct > 5:
                ax.text(xi, bot + pct / 2, f'{pct:.0f}%',
                        ha='center', va='center', fontsize=8,
                        color='white', fontweight='bold')
        bottoms += np.array(pcts)

    ax.set_xticks(x)
    ax.set_xticklabels(algos, fontsize=12)
    ax.set_ylabel('Percentage of episodes (%)', fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_title('Episode Termination Reason Breakdown', fontsize=12)
    ax.legend(loc='upper right', fontsize=8,
              bbox_to_anchor=(1.22, 1), borderaxespad=0)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


# ============================================================================
# Plot: delivery time comparison (drone vs ambulance)
# ============================================================================

def plot_delivery_time_comparison(results_dict: dict, out_path: str) -> None:
    """Violin plot comparing successful drone delivery times vs ambulance baseline."""
    rng = np.random.default_rng(42)
    amb_samples = rng.normal(AMBULANCE_MEAN_S / 60, AMBULANCE_STD_S / 60, 1000)

    labels  = ['Ambulance\nbaseline']
    data    = [amb_samples]
    colors  = ['#95a5a6']

    for algo in results_dict:
        times = [r['total_time_s'] / 60
                 for r in results_dict[algo] if r['success']]
        if times:
            labels.append(f'{algo}\n(successful)')
            data.append(times)
            colors.append(ALGO_COLORS[algo])

    fig, ax = plt.subplots(figsize=(9, 6))
    parts = ax.violinplot(data, positions=range(len(labels)),
                          showmeans=True, showmedians=True)
    for body, color in zip(parts['bodies'], colors):
        body.set_facecolor(color)
        body.set_alpha(0.65)

    ax.axhline(AMBULANCE_MEAN_S / 60, color='grey', linestyle='--',
               linewidth=1.2, alpha=0.6,
               label=f'Ambulance mean ({AMBULANCE_MEAN_S/60:.0f} min)')

    for i, (label, vals) in enumerate(zip(labels, data)):
        mu = np.mean(vals)
        sd = np.std(vals)
        ax.text(i + 0.05, ax.get_ylim()[1] * 0.02 if ax.get_ylim()[1] > 0 else 0,
                f'μ={mu:.1f}\nσ={sd:.1f}',
                ha='left', va='bottom', fontsize=8, color='#333333')

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Delivery time (minutes)', fontsize=10)
    ax.set_title(
        'Delivery Time: Drone vs Ambulance Baseline\n'
        f'Ambulance: μ={AMBULANCE_MEAN_S/60:.0f} min, σ={AMBULANCE_STD_S/60:.0f} min',
        fontsize=11,
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


# ============================================================================
# Text: comparison table
# ============================================================================

def plot_altitude_profile(results_dict: dict, out_path: str,
                          max_trajs: int = 20) -> None:
    """Altitude vs simulated time for sampled trajectories, coloured by outcome."""
    from aorva_env import MIN_ALT_M, MAX_ALT_M, DT
    algos = list(results_dict.keys())
    fig, axes = plt.subplots(1, len(algos), figsize=(10 * len(algos), 5), squeeze=False)

    for col, algo in enumerate(algos):
        ax = axes[0][col]
        results = results_dict[algo]

        success_plotted = failure_plotted = False
        for r in results[:max_trajs]:
            traj = r.get("trajectory", [])
            if not traj:
                continue
            alts  = [p[2] for p in traj]
            times = [i * DT for i in range(len(alts))]
            color = "limegreen" if r["success"] else "tomato"
            label = None
            if r["success"] and not success_plotted:
                label = "Success"; success_plotted = True
            elif not r["success"] and not failure_plotted:
                label = "Failure"; failure_plotted = True
            ax.plot(times, alts, color=color, linewidth=0.7, alpha=0.45, label=label)

        ax.axhline(MIN_ALT_M, color="red",    linestyle="--", linewidth=1.4,
                   label=f"Min alt ({MIN_ALT_M:.0f} m)")
        ax.axhline(MAX_ALT_M, color="orange", linestyle="--", linewidth=1.4,
                   label=f"Max alt ({MAX_ALT_M:.0f} m)")
        ax.axhspan(0, MIN_ALT_M, alpha=0.08, color="red")
        ax.axhspan(MAX_ALT_M, MAX_ALT_M + 30, alpha=0.08, color="orange")

        ax.set_xlabel("Simulated time (s)", fontsize=9)
        ax.set_ylabel("Altitude (m)", fontsize=9)
        ax.set_title(f"{algo} — Altitude profile over time"
                     f"(n={min(len(results), max_trajs)} trajectories, "
                     f"green=success, red=failure)", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Altitude Profile — Safe Band Compliance",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def print_comparison_table(metrics_dict: dict, out_path: str) -> None:
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
        row("Mean episode reward",
            m_ppo.get('mean_reward', float('nan')),
            m_sac.get('mean_reward', float('nan')),
            None, '.1f'),
        sep,
    ]

    sigma_amb = AMBULANCE_STD_S / 60
    for algo, m in [('PPO', m_ppo), ('SAC', m_sac)]:
        sigma_d = m.get('std_time_s', float('nan')) / 60
        delta_s = sigma_amb - sigma_d if not np.isnan(sigma_d) else float('nan')
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


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AORVA Step 6 - Evaluate trained RL agents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--ppo-only', action='store_true')
    parser.add_argument('--sac-only', action='store_true')
    parser.add_argument('--stochastic', action='store_true')
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

    env = AORVAEnv()

    results_by_algo  = {}
    metrics_by_algo  = {}

    for algo in algos_to_eval:
        model_path = _find_model(algo)
        if model_path is None:
            print(f"\n{algo}: no trained model found. "
                  f"Run scripts/04_train_agents.py {algo.lower()} first.")
            continue

        print(f"\n-- {algo} ---------------------------------")
        print(f"   Model: {model_path}")
        model, vec_norm = _load_model(algo, model_path, env)

        print(f"   Running {args.episodes} episodes "
              f"({'deterministic' if deterministic else 'stochastic'})...")
        results = run_episodes(model, env, args.episodes,
                               vec_norm=vec_norm, deterministic=deterministic)
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
        print(f"     Mean episode reward: {m['mean_reward']:.1f} ± {m['std_reward']:.1f}")

    if not results_by_algo:
        print("\nNo models evaluated. Train at least one agent first.")
        sys.exit(1)

    print("\n-- Generating Phase 4 visualisations --")

    plot_training_curves('outputs/eval_training_curves.png')

    if metrics_by_algo:
        plot_checkpoint_timing(metrics_by_algo, env,
                                'outputs/eval_checkpoint_timing.png')

    if results_by_algo:
        plot_trajectory_map(results_by_algo, env,
                             'outputs/eval_trajectory_map.png')
        plot_reward_distribution(results_by_algo,
                                  'outputs/eval_reward_distribution.png')
        plot_termination_breakdown(results_by_algo,
                                    'outputs/eval_termination_breakdown.png')
        plot_delivery_time_comparison(results_by_algo,
                                       'outputs/eval_delivery_time.png')
        plot_altitude_profile(results_by_algo,
                              'outputs/eval_altitude_profile.png')

    print_comparison_table(metrics_by_algo,
                            'outputs/eval_comparison_table.txt')

    print("\nStep 6 complete.")
    print("\nOutputs written to outputs/:")
    print("  eval_training_curves.png       - reward per episode, markers every 10")
    print("  eval_checkpoint_timing.png     - actual vs target arrival times")
    print("  eval_trajectory_map.png        - drone paths (green=success, red=fail)")
    print("  eval_reward_distribution.png   - violin plot of episode rewards")
    print("  eval_termination_breakdown.png - why episodes ended (stacked bar)")
    print("  eval_delivery_time.png         - drone vs ambulance delivery time")
    print("  eval_comparison_table.txt      - PPO vs SAC vs Ambulance table")


if __name__ == "__main__":
    main()
