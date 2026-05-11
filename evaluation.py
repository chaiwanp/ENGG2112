"""
evaluate_agents.py

Evaluate trained PPO and SAC agents on AORVAEnv.

Usage
-----
    python evaluate_agents.py
    python evaluate_agents.py --episodes 50
    python evaluate_agents.py --ppo-only
    python evaluate_agents.py --sac-only

Outputs
-------
    outputs/evaluation_results.csv
    outputs/evaluation_summary.png
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import PPO, SAC

from aorva_env import AORVAEnv

MODEL_DIR = Path('models')
OUT_DIR   = Path('outputs')
OUT_DIR.mkdir(exist_ok=True)


# -----------------------------------------------------------------------
# Single episode runner
# -----------------------------------------------------------------------
def run_episode(model, env, deterministic=True) -> dict:
    obs, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0.0

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

    deviations      = info.get('checkpoint_deviations', [])
    checkpoints_hit = info.get('checkpoints_passed', 0)
    battery_left    = info.get('battery', 0.0)
    sim_time        = info.get('sim_time', 0.0)

    reached_goal     = checkpoints_hit >= 14
    battery_dead     = (not reached_goal) and battery_left <= 0.0
    safety_violation = (not reached_goal) and (not battery_dead)

    rmse = float(np.sqrt(np.mean(np.array(deviations) ** 2))) if deviations else np.nan

    return {
        'total_reward':        total_reward,
        'reached_goal':        reached_goal,
        'checkpoints_passed':  checkpoints_hit,
        'delivery_time_s':     sim_time if reached_goal else np.nan,
        'rmse_deviation_s':    rmse,
        'safety_violation':    safety_violation,
        'battery_dead':        battery_dead,
        'battery_remaining':   battery_left,
        'urgency':             info.get('urgency', 1.0),
    }


# -----------------------------------------------------------------------
# Evaluate one agent over N episodes
# -----------------------------------------------------------------------
def evaluate_agent(model, label: str, n_episodes: int) -> pd.DataFrame:
    env = AORVAEnv()
    records = []

    print(f"\nEvaluating {label} over {n_episodes} episodes...")
    for ep in range(n_episodes):
        result = run_episode(model, env)
        result['agent']   = label
        result['episode'] = ep
        records.append(result)

        if (ep + 1) % 10 == 0:
            completed = sum(r['reached_goal'] for r in records)
            print(f"  ep {ep+1:3d}/{n_episodes} | "
                  f"goal rate: {completed/(ep+1)*100:.0f}% | "
                  f"mean reward: {np.mean([r['total_reward'] for r in records]):.1f}")

    env.close()
    return pd.DataFrame(records)


# -----------------------------------------------------------------------
# Print summary table
# -----------------------------------------------------------------------
def print_summary(all_df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    rows = []
    for agent in all_df['agent'].unique():
        sub = all_df[all_df['agent'] == agent]
        rows.append({
            'Agent':                agent,
            'Goal rate (%)':        f"{sub['reached_goal'].mean()*100:.1f}",
            'Mean delivery (s)':    f"{sub['delivery_time_s'].mean():.1f}",
            'Std delivery (s)':     f"{sub['delivery_time_s'].std():.1f}",
            'RMSE deviation (s)':   f"{sub['rmse_deviation_s'].mean():.1f}",
            'Safety violation (%)': f"{sub['safety_violation'].mean()*100:.1f}",
            'Battery dead (%)':     f"{sub['battery_dead'].mean()*100:.1f}",
            'Mean reward':          f"{sub['total_reward'].mean():.1f}",
        })

    print(pd.DataFrame(rows).set_index('Agent').to_string())
    print("=" * 60)


# -----------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------
def plot_results(all_df: pd.DataFrame) -> None:
    agents = all_df['agent'].unique()
    colors = {'PPO': '#2196F3', 'SAC': '#FF5722'}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('AORVA Agent Evaluation', fontsize=14, fontweight='bold')

    # Delivery time distribution
    ax = axes[0, 0]
    for agent in agents:
        times = all_df[all_df['agent'] == agent]['delivery_time_s'].dropna()
        if len(times) > 0:
            ax.hist(times, bins=20, alpha=0.6, label=agent,
                    color=colors.get(agent, '#888'))
    ax.set_xlabel('Delivery time (s)')
    ax.set_ylabel('Count')
    ax.set_title('Delivery Time Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Checkpoint RMSE boxplot
    ax = axes[0, 1]
    rmse_data = [all_df[all_df['agent'] == a]['rmse_deviation_s'].dropna().values
                 for a in agents]
    bp = ax.boxplot(rmse_data, labels=agents, patch_artist=True)
    for patch, agent in zip(bp['boxes'], agents):
        patch.set_facecolor(colors.get(agent, '#888'))
        patch.set_alpha(0.7)
    ax.set_ylabel('RMSE (s)')
    ax.set_title('Checkpoint Arrival RMSE')
    ax.grid(True, alpha=0.3)

    # Goal completion rate
    ax = axes[1, 0]
    goal_rates = [all_df[all_df['agent'] == a]['reached_goal'].mean() * 100
                  for a in agents]
    bars = ax.bar(agents, goal_rates,
                  color=[colors.get(a, '#888') for a in agents], alpha=0.8)
    ax.set_ylabel('Goal completion (%)')
    ax.set_title('Goal Completion Rate')
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, goal_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=11)

    # Delivery time std
    ax = axes[1, 1]
    std_vals = [all_df[all_df['agent'] == a]['delivery_time_s'].dropna().std()
                for a in agents]
    bars = ax.bar(agents, std_vals,
                  color=[colors.get(a, '#888') for a in agents], alpha=0.8)
    ax.set_ylabel('Std dev of delivery time (s)')
    ax.set_title('Delivery Variance (lower = better)')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, std_vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.0f}s', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    out_path = OUT_DIR / 'evaluation_summary.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"\nSaved to {out_path}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=30)
    parser.add_argument('--ppo-only', action='store_true')
    parser.add_argument('--sac-only', action='store_true')
    args = parser.parse_args()

    all_dfs = []

    if not args.sac_only:
        ppo_path = MODEL_DIR / 'ppo_aorva_final.zip'
        if not ppo_path.exists():
            ppo_path = MODEL_DIR / 'ppo_best' / 'best_model.zip'
        if ppo_path.exists():
            print(f"Loading PPO from {ppo_path}")
            ppo_df = evaluate_agent(PPO.load(str(ppo_path)), 'PPO', args.episodes)
            all_dfs.append(ppo_df)
        else:
            print("PPO model not found — train first with _evaluate_agents.py")

    if not args.ppo_only:
        sac_path = MODEL_DIR / 'sac_aorva_final.zip'
        if not sac_path.exists():
            sac_path = MODEL_DIR / 'sac_best' / 'best_model.zip'
        if sac_path.exists():
            print(f"Loading SAC from {sac_path}")
            sac_df = evaluate_agent(SAC.load(str(sac_path)), 'SAC', args.episodes)
            all_dfs.append(sac_df)
        else:
            print("SAC model not found — train first with _evaluate_agents.py")

    if not all_dfs:
        print("No models found.")
        return

    all_df = pd.concat(all_dfs, ignore_index=True)
    all_df.to_csv(OUT_DIR / 'evaluation_results.csv', index=False)
    print(f"\nRaw results saved to {OUT_DIR / 'evaluation_results.csv'}")

    print_summary(all_df)
    plot_results(all_df)


if __name__ == '__main__':
    main()