"""
evaluate_agents.py

Week 11 evaluation framework. Loads trained PPO and SAC policies, runs
each through N episodes with varied wind conditions, and computes the
three evaluation criteria committed to in the project proposal:

    A. RMSE of arrival time deviation at checkpoints
    B. Safety violation rate (target: 0%)
    C. Delivery variance reduction vs ambulance baseline

Outputs a JSON results file, summary printout, and four comparison
plots ready to drop into the final report.

Usage
-----
    python evaluate_agents.py                          # default N=100
    python evaluate_agents.py --n 200
    python evaluate_agents.py --ppo-only
    python evaluate_agents.py --ambulance ambulance_data.csv

Why N=100 by default
--------------------
At N=100 with zero observed safety violations, the upper 95% confidence
bound on the true violation rate is approximately 3% (binomial). That's
a defensible "approaching zero" claim for a proof-of-concept. Increasing
to N=300 tightens that bound to ~1%. Any smaller and the safety-rate
claim isn't statistically meaningful.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, SAC

from aorva_env import AORVAEnv, GOAL_RADIUS_M, MIN_ALT_M


# ======================================================================
# Result containers
# ======================================================================
@dataclass
class FlightResult:
    """Outcome of a single evaluation flight."""
    episode_idx: int
    seed: int
    success: bool
    termination_reason: str       # 'goal' | 'crash' | 'low_alt' | 'high_alt'
                                  # | 'out_of_bounds' | 'battery' | 'timeout'
    flight_time_s: float
    checkpoint_deviations_s: list[float]
    checkpoints_passed: int
    final_battery: float
    safety_violation: bool        # True if crash or low altitude


@dataclass
class AgentResults:
    """Aggregated results for a single trained agent."""
    name: str
    flights: list[FlightResult] = field(default_factory=list)

    @property
    def n_total(self) -> int:
        return len(self.flights)

    @property
    def n_successful(self) -> int:
        return sum(1 for f in self.flights if f.success)

    @property
    def n_safety_violations(self) -> int:
        return sum(1 for f in self.flights if f.safety_violation)

    @property
    def successful_flight_times(self) -> np.ndarray:
        return np.array([f.flight_time_s for f in self.flights if f.success])

    @property
    def all_checkpoint_deviations(self) -> np.ndarray:
        devs = []
        for f in self.flights:
            if f.success:
                devs.extend(f.checkpoint_deviations_s)
        return np.array(devs)


# ======================================================================
# Ambulance baseline
# ======================================================================
def ambulance_baseline(n_samples: int, csv_path: Optional[str] = None,
                       seed: int = 42) -> np.ndarray:
    """
    Return n_samples of ambulance travel times for Westmead -> Liverpool.

    If csv_path provided, loads from a column called 'travel_time_s'.
    Otherwise generates a synthetic distribution based on the TomTom
    Sydney Traffic Index referenced in the project proposal:
        Peak hour:    Sydney mean speed ~20 km/h, route ~30 km by road
                      -> mean 90 min, but high variance (sigma ~15 min).
        Off-peak:     ~30-35 km/h -> mean 55 min, sigma ~5 min.
    50/50 mix of the two.
    """
    if csv_path is not None:
        import pandas as pd
        df = pd.read_csv(csv_path)
        if 'travel_time_s' not in df.columns:
            raise ValueError("Ambulance CSV must have a 'travel_time_s' column")
        return df['travel_time_s'].sample(
            n_samples, replace=True, random_state=seed
        ).values

    # Synthetic baseline calibrated to TomTom Sydney traffic data
    rng = np.random.default_rng(seed)
    is_peak = rng.random(n_samples) < 0.5
    peak_times = rng.normal(loc=5400, scale=900, size=n_samples)      # 90 +/- 15 min
    offpeak_times = rng.normal(loc=3300, scale=300, size=n_samples)   # 55 +/- 5 min
    times = np.where(is_peak, peak_times, offpeak_times)
    return np.clip(times, 1500, 9000)   # bound to physically plausible range


# ======================================================================
# Single-flight evaluation
# ======================================================================
def classify_termination(env: AORVAEnv, terminated: bool, truncated: bool
                         ) -> tuple[str, bool, bool]:
    """
    Inspect final env state to determine why the episode ended.

    Returns
    -------
    reason : str         One of 'goal', 'crash', 'low_alt', 'high_alt',
                         'out_of_bounds', 'battery', 'timeout'.
    success : bool       True iff goal reached.
    safety_violation : bool   True if crash or low altitude (per
                              proposal Section 3.B definition).
    """
    if truncated and not terminated:
        return 'timeout', False, False

    pos = env.pos
    if np.linalg.norm(pos - env._goal_world) < GOAL_RADIUS_M:
        return 'goal', True, False

    ix, iy, iz = env._world_to_voxel(pos)
    if env.voxel_grid.grid[ix, iy, iz] == 1:
        return 'crash', False, True

    if pos[2] < MIN_ALT_M:
        return 'low_alt', False, True

    if pos[2] > 300.0:
        return 'high_alt', False, False

    if (pos[0] < 0 or pos[0] > env.voxel_grid.width_m or
            pos[1] < 0 or pos[1] > env.voxel_grid.length_m):
        return 'out_of_bounds', False, False

    if env.battery <= 0.0:
        return 'battery', False, False

    return 'unknown', False, False


def run_single_flight(model, env: AORVAEnv, seed: int, episode_idx: int
                      ) -> FlightResult:
    """Run one deterministic flight with the given seed."""
    obs, info = env.reset(seed=seed)
    terminated = truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

    reason, success, safety_violation = classify_termination(
        env, terminated, truncated
    )

    return FlightResult(
        episode_idx=episode_idx,
        seed=seed,
        success=success,
        termination_reason=reason,
        flight_time_s=info['sim_time'],
        checkpoint_deviations_s=info['checkpoint_deviations'],
        checkpoints_passed=info['checkpoints_passed'],
        final_battery=info['battery'],
        safety_violation=safety_violation,
    )


def evaluate_agent(model, name: str, n_episodes: int, seed_base: int = 1000
                   ) -> AgentResults:
    """Run an agent through N episodes with varied wind, return aggregated."""
    print(f"\nEvaluating {name} over {n_episodes} episodes...")
    env = AORVAEnv()
    results = AgentResults(name=name)

    for i in range(n_episodes):
        seed = seed_base + i
        flight = run_single_flight(model, env, seed, i)
        results.flights.append(flight)

        # Progress indicator every 10 episodes
        if (i + 1) % 10 == 0:
            n_success = sum(1 for f in results.flights if f.success)
            n_safety = sum(1 for f in results.flights if f.safety_violation)
            print(f"  [{i + 1}/{n_episodes}]  success: {n_success}  "
                  f"safety violations: {n_safety}")

    return results


# ======================================================================
# Metric computation (proposal Section 3)
# ======================================================================
def compute_metrics(agent: AgentResults, ambulance_times: np.ndarray) -> dict:
    """
    Compute the three proposal metrics for a single agent.

    A. RMSE of arrival time deviation at checkpoints (over successful flights)
    B. Safety violation rate (over all flights)
    C. Delivery variance reduction vs ambulance baseline
    """
    # --- A. RMSE of arrival time deviation ---
    devs = agent.all_checkpoint_deviations
    if len(devs) > 0:
        rmse_s = float(np.sqrt(np.mean(devs ** 2)))
        mean_abs_dev = float(np.mean(np.abs(devs)))
        max_abs_dev = float(np.max(np.abs(devs)))
    else:
        rmse_s = mean_abs_dev = max_abs_dev = float('nan')

    # --- B. Safety violation rate ---
    safety_rate = agent.n_safety_violations / agent.n_total
    # Wilson 95% upper bound for binomial proportion
    z = 1.96
    p_hat = safety_rate
    n = agent.n_total
    if n > 0:
        denom = 1 + z**2 / n
        centre = p_hat + z**2 / (2 * n)
        radius = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
        wilson_upper = (centre + radius) / denom
    else:
        wilson_upper = float('nan')

    # --- C. Delivery variance reduction vs ambulance ---
    drone_times = agent.successful_flight_times
    if len(drone_times) >= 2:
        sigma_drone = float(np.std(drone_times, ddof=1))
        mean_drone = float(np.mean(drone_times))
    else:
        sigma_drone = mean_drone = float('nan')

    sigma_ambulance = float(np.std(ambulance_times, ddof=1))
    mean_ambulance = float(np.mean(ambulance_times))

    if not np.isnan(sigma_drone) and sigma_ambulance > 0:
        variance_reduction = (sigma_ambulance - sigma_drone) / sigma_ambulance
        time_savings_pct = (mean_ambulance - mean_drone) / mean_ambulance
    else:
        variance_reduction = time_savings_pct = float('nan')

    # --- Termination breakdown ---
    breakdown = {}
    for f in agent.flights:
        breakdown[f.termination_reason] = breakdown.get(f.termination_reason, 0) + 1

    return {
        'agent': agent.name,
        'n_total': agent.n_total,
        'n_successful': agent.n_successful,
        'success_rate': agent.n_successful / agent.n_total,
        # A. Time accuracy
        'rmse_arrival_deviation_s': rmse_s,
        'mean_abs_deviation_s': mean_abs_dev,
        'max_abs_deviation_s': max_abs_dev,
        # B. Safety
        'safety_violation_rate': safety_rate,
        'safety_violation_rate_95_upper': float(wilson_upper),
        'n_safety_violations': agent.n_safety_violations,
        # C. Variance
        'sigma_drone_s': sigma_drone,
        'sigma_ambulance_s': sigma_ambulance,
        'mean_drone_s': mean_drone,
        'mean_ambulance_s': mean_ambulance,
        'variance_reduction': variance_reduction,
        'time_savings_pct': time_savings_pct,
        # Diagnostics
        'termination_breakdown': breakdown,
    }


# ======================================================================
# Plotting
# ======================================================================
def plot_results(all_metrics: list[dict],
                 results_by_agent: dict[str, AgentResults],
                 ambulance_times: np.ndarray,
                 outdir: Path) -> None:
    """Four plots for the report."""
    outdir.mkdir(parents=True, exist_ok=True)
    agent_names = [m['agent'] for m in all_metrics]
    colors = {'PPO': '#1f77b4', 'SAC': '#ff7f0e', 'Ambulance': '#7f7f7f'}

    # --- Plot 1: Flight time distributions ---
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(500, 6000, 40)
    for name in agent_names:
        times = results_by_agent[name].successful_flight_times
        if len(times) > 0:
            ax.hist(times, bins=bins, alpha=0.6,
                    label=f'{name} (n={len(times)})',
                    color=colors.get(name, 'C0'), edgecolor='black')
    ax.hist(ambulance_times, bins=bins, alpha=0.5, label='Ambulance baseline',
            color=colors['Ambulance'], edgecolor='black')
    ax.set_xlabel('Flight time (s)')
    ax.set_ylabel('Count')
    ax.set_title('Delivery time distributions (successful flights only)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / 'flight_time_distributions.png',
                dpi=200, bbox_inches='tight')
    plt.close()

    # --- Plot 2: Checkpoint deviation box plots ---
    fig, ax = plt.subplots(figsize=(10, 6))
    data, labels = [], []
    for name in agent_names:
        devs = results_by_agent[name].all_checkpoint_deviations
        if len(devs) > 0:
            data.append(devs)
            labels.append(f'{name}\n(n={len(devs)})')
    if data:
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, name in zip(bp['boxes'], agent_names):
            patch.set_facecolor(colors.get(name, 'C0'))
            patch.set_alpha(0.7)
    ax.axhline(0, color='red', linestyle='--', alpha=0.5,
               label='On-time (deviation = 0)')
    ax.set_ylabel('Arrival time deviation (s)')
    ax.set_title('Checkpoint arrival time deviations')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(outdir / 'checkpoint_deviations.png',
                dpi=200, bbox_inches='tight')
    plt.close()

    # --- Plot 3: Safety violation comparison ---
    fig, ax = plt.subplots(figsize=(8, 6))
    rates = [m['safety_violation_rate'] for m in all_metrics]
    upper = [m['safety_violation_rate_95_upper'] for m in all_metrics]
    err = [u - r for u, r in zip(upper, rates)]
    bars = ax.bar(agent_names, rates,
                  yerr=[[0] * len(rates), err],
                  capsize=10,
                  color=[colors.get(n, 'C0') for n in agent_names],
                  edgecolor='black')
    for bar, rate, m in zip(bars, rates, all_metrics):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f'{rate:.1%}\n({m["n_safety_violations"]}/{m["n_total"]})',
                ha='center', va='bottom', fontsize=10)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel('Safety violation rate')
    ax.set_title('Safety violation rate (target: 0%)\n'
                 'Error bars: 95% Wilson upper bound')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(outdir / 'safety_violation_rates.png',
                dpi=200, bbox_inches='tight')
    plt.close()

    # --- Plot 4: Variance comparison ---
    fig, ax = plt.subplots(figsize=(8, 6))
    sigmas = [m['sigma_ambulance_s'] for m in all_metrics] + \
             [m['sigma_drone_s'] for m in all_metrics if not np.isnan(m['sigma_drone_s'])]
    labels = ['Ambulance'] + [m['agent'] for m in all_metrics
                              if not np.isnan(m['sigma_drone_s'])]
    plot_sigmas = [all_metrics[0]['sigma_ambulance_s']] + [
        m['sigma_drone_s'] for m in all_metrics if not np.isnan(m['sigma_drone_s'])
    ]
    bar_colors = [colors['Ambulance']] + [
        colors.get(m['agent'], 'C0') for m in all_metrics
        if not np.isnan(m['sigma_drone_s'])
    ]
    bars = ax.bar(labels, plot_sigmas, color=bar_colors, edgecolor='black')
    for bar, sigma in zip(bars, plot_sigmas):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(plot_sigmas) * 0.01,
                f'{sigma:.0f}s', ha='center', va='bottom', fontsize=11)
    ax.set_ylabel('Standard deviation of delivery time (s)')
    ax.set_title('Delivery time consistency: lower sigma is better')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(outdir / 'variance_comparison.png',
                dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\nPlots saved to {outdir}/")


# ======================================================================
# Reporting
# ======================================================================
def print_summary(all_metrics: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    for m in all_metrics:
        print(f"\n--- {m['agent']} ---")
        print(f"  Episodes:          {m['n_total']}")
        print(f"  Success rate:      {m['success_rate']:.1%}  "
              f"({m['n_successful']}/{m['n_total']})")

        print(f"\n  CRITERION A: Arrival time accuracy")
        print(f"    RMSE (checkpoint deviations):  "
              f"{m['rmse_arrival_deviation_s']:.2f} s")
        print(f"    Mean |deviation|:              "
              f"{m['mean_abs_deviation_s']:.2f} s")
        print(f"    Max |deviation|:               "
              f"{m['max_abs_deviation_s']:.2f} s")

        print(f"\n  CRITERION B: Safety violation rate (target: 0%)")
        print(f"    Observed rate:                 "
              f"{m['safety_violation_rate']:.2%}  "
              f"({m['n_safety_violations']}/{m['n_total']})")
        print(f"    95% upper bound:               "
              f"{m['safety_violation_rate_95_upper']:.2%}")

        print(f"\n  CRITERION C: Variance reduction vs ambulance")
        print(f"    Drone sigma:                   "
              f"{m['sigma_drone_s']:.1f} s")
        print(f"    Ambulance sigma:               "
              f"{m['sigma_ambulance_s']:.1f} s")
        print(f"    Variance reduction (Δσ):       "
              f"{m['variance_reduction']:.1%}")
        print(f"    Mean time savings:             "
              f"{m['time_savings_pct']:.1%}")

        print(f"\n  Termination breakdown:")
        for reason, count in sorted(m['termination_breakdown'].items(),
                                    key=lambda kv: -kv[1]):
            pct = count / m['n_total']
            print(f"    {reason:20s} {count:3d}  ({pct:.1%})")

    print("\n" + "=" * 70)


def serialise_results(results_by_agent: dict[str, AgentResults]) -> dict:
    """Convert AgentResults objects to JSON-friendly dicts."""
    return {
        name: {
            'name': r.name,
            'flights': [asdict(f) for f in r.flights],
        }
        for name, r in results_by_agent.items()
    }


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=100,
                        help='Number of evaluation episodes per agent')
    parser.add_argument('--ppo-model', default='models/ppo_aorva_final.zip')
    parser.add_argument('--sac-model', default='models/sac_aorva_final.zip')
    parser.add_argument('--ppo-only', action='store_true')
    parser.add_argument('--sac-only', action='store_true')
    parser.add_argument('--ambulance', default=None,
                        help='CSV with travel_time_s column for ambulance baseline')
    parser.add_argument('--outdir', default='outputs/evaluation')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Ambulance baseline ---
    print(f"Generating ambulance baseline (n={args.n})...")
    ambulance_times = ambulance_baseline(args.n, csv_path=args.ambulance)
    print(f"  Mean: {ambulance_times.mean():.0f}s ({ambulance_times.mean() / 60:.1f} min)")
    print(f"  Std:  {ambulance_times.std(ddof=1):.0f}s")

    # --- Evaluate agents ---
    results_by_agent: dict[str, AgentResults] = {}
    all_metrics: list[dict] = []

    if not args.sac_only:
        if not Path(args.ppo_model).exists():
            print(f"WARNING: PPO model not found at {args.ppo_model}, skipping")
        else:
            ppo = PPO.load(args.ppo_model)
            ppo_results = evaluate_agent(ppo, 'PPO', args.n)
            results_by_agent['PPO'] = ppo_results
            all_metrics.append(compute_metrics(ppo_results, ambulance_times))

    if not args.ppo_only:
        if not Path(args.sac_model).exists():
            print(f"WARNING: SAC model not found at {args.sac_model}, skipping")
        else:
            sac = SAC.load(args.sac_model)
            sac_results = evaluate_agent(sac, 'SAC', args.n)
            results_by_agent['SAC'] = sac_results
            all_metrics.append(compute_metrics(sac_results, ambulance_times))

    if not all_metrics:
        print("\nNo agents evaluated. Train at least one model first.")
        return

    # --- Output ---
    print_summary(all_metrics)
    plot_results(all_metrics, results_by_agent, ambulance_times, outdir)

    # JSON dump
    output = {
        'config': {
            'n_episodes': args.n,
            'ambulance_source': args.ambulance or 'synthetic_TomTom_calibrated',
            'ambulance_mean_s': float(ambulance_times.mean()),
            'ambulance_sigma_s': float(ambulance_times.std(ddof=1)),
        },
        'metrics': all_metrics,
        'flights': serialise_results(results_by_agent),
    }
    with open(outdir / 'evaluation_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nFull results saved to {outdir / 'evaluation_results.json'}")


if __name__ == '__main__':
    main()
