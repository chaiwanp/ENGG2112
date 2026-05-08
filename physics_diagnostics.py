"""
physics_diagnostics.py

Sanity-check the AORVA simulation physics before retraining. Catches
the silent bugs (coordinate flips, unrealistic wind magnitudes, broken
energy budgets) that no amount of reward tuning can fix.

Runs four checks:
    1. Velocity step response  -- confirms first-order dynamics behave
                                  as intended
    2. Wind field magnitudes   -- confirms log-law output is realistic
                                  across altitudes
    3. Energy budget           -- confirms a complete naive flight
                                  drains a sensible fraction of battery
    4. Coordinate consistency  -- confirms lat/lon <-> world <-> voxel
                                  transforms round-trip cleanly

If anything fails, fix it before retraining. Print output is structured
so you can paste it directly into a progress report.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from aorva_env import AORVAEnv, V_MAX, DT, TAU_V


# ======================================================================
# 1. Velocity step response
# ======================================================================
def check_velocity_response():
    print("\n" + "=" * 60)
    print("CHECK 1: Velocity step response")
    print("=" * 60)

    env = AORVAEnv()
    env.reset(seed=42)

    # Command full velocity in +x for 3 seconds, then hold
    velocities = []
    times = []
    for step in range(60):   # 6 seconds
        action = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        env.step(action)
        velocities.append(env.vel.copy())
        times.append(env.sim_time)

    velocities = np.array(velocities)

    # Theoretical first-order step response: v(t) = V_MAX * (1 - exp(-t/tau))
    t = np.array(times)
    expected = V_MAX * (1.0 - np.exp(-t / TAU_V))

    achieved_at_tau = velocities[int(TAU_V / DT) - 1, 0]
    expected_at_tau = V_MAX * (1.0 - np.exp(-1.0))   # 63.2% of V_MAX

    print(f"  V_MAX = {V_MAX} m/s, tau = {TAU_V} s")
    print(f"  At t=tau ({TAU_V}s):  measured = {achieved_at_tau:.2f} m/s  "
          f"expected = {expected_at_tau:.2f} m/s")
    print(f"  At t=3*tau ({3*TAU_V}s): measured = "
          f"{velocities[int(3*TAU_V/DT) - 1, 0]:.2f} m/s  "
          f"(should be ~{V_MAX*0.95:.1f} m/s)")

    err = abs(achieved_at_tau - expected_at_tau) / expected_at_tau
    if err < 0.05:
        print(f"  PASS (error {err:.1%})")
    else:
        print(f"  FAIL (error {err:.1%}) -- velocity dynamics are off")

    # Save plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times, velocities[:, 0], 'b-', linewidth=2, label='Measured v_x')
    ax.plot(t, expected, 'r--', linewidth=2, label='First-order theoretical')
    ax.axvline(TAU_V, color='gray', linestyle=':', alpha=0.7,
               label=f'tau = {TAU_V}s')
    ax.axhline(V_MAX * (1 - np.exp(-1)), color='gray', linestyle=':', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Velocity step response: command = [V_MAX, 0, 0]')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig('outputs/diagnostic_velocity_response.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print("  Saved outputs/diagnostic_velocity_response.png")


# ======================================================================
# 2. Wind field magnitudes
# ======================================================================
def check_wind_magnitudes():
    print("\n" + "=" * 60)
    print("CHECK 2: Wind field magnitudes")
    print("=" * 60)

    env = AORVAEnv()
    env.reset(seed=42)
    vg = env.voxel_grid
    wf = env.wind_field

    # Sample wind across altitudes at the centre of the grid
    cx, cy = vg.nx // 2, vg.ny // 2

    print(f"  Sampling at grid centre (x={cx}, y={cy}) across altitudes:")
    print(f"  {'Altitude':>10}  {'|wind|':>9}")
    altitudes = [10, 50, 100, 150, 200, 300, 400]
    speeds = []
    for alt_m in altitudes:
        iz = min(int(alt_m / vg.voxel_size_m), vg.nz - 1)
        u, v, w = wf.get_wind_at_position(cx, cy, iz)
        speed = np.sqrt(u**2 + v**2 + w**2)
        speeds.append(speed)
        print(f"  {alt_m:>7} m   {speed:>6.2f} m/s")

    # Plausibility checks
    issues = []
    if max(speeds) > 30:
        issues.append("max wind > 30 m/s (108 km/h) -- suspicious for typical day")
    if max(speeds) < 1:
        issues.append("max wind < 1 m/s -- wind field may be near-zero")
    if speeds[-1] < speeds[0]:
        issues.append("wind decreases with altitude -- log-law has wrong sign")

    if issues:
        print("  ISSUES FOUND:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  PASS  Wind magnitudes are physically plausible.")

    # Plot vertical profile
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.plot(speeds, altitudes, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Wind speed (m/s)')
    ax.set_ylabel('Altitude (m)')
    ax.set_title('Vertical wind profile at grid centre')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/diagnostic_wind_profile.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print("  Saved outputs/diagnostic_wind_profile.png")


# ======================================================================
# 3. Energy budget
# ======================================================================
def check_energy_budget():
    print("\n" + "=" * 60)
    print("CHECK 3: Energy budget across a full naive flight")
    print("=" * 60)

    env = AORVAEnv()
    obs, info = env.reset(seed=42)

    initial_battery = env.battery
    battery_log = [initial_battery]

    for step in range(20_000):
        direction = env._goal_world - env.pos
        norm = np.linalg.norm(direction)
        action = direction / max(norm, 1e-6)
        obs, r, terminated, truncated, info = env.step(action)
        battery_log.append(env.battery)
        if terminated or truncated:
            break

    battery_used = initial_battery - env.battery
    flight_time = env.sim_time

    print(f"  Flight duration:     {flight_time:.1f} s ({flight_time/60:.1f} min)")
    print(f"  Initial battery:     {initial_battery:.2%}")
    print(f"  Final battery:       {env.battery:.2%}")
    print(f"  Battery consumed:    {battery_used:.2%}")
    print(f"  Burn rate:           {battery_used / flight_time * 100:.3f} %/s")

    if 0.10 < battery_used < 0.50:
        print(f"  PASS  ({battery_used:.0%} consumed for full flight is realistic)")
    elif battery_used >= 0.50:
        print(f"  WARN  ({battery_used:.0%} too aggressive -- "
              f"reduce energy term in reward")
    else:
        print(f"  WARN  ({battery_used:.0%} too soft -- "
              f"energy term has no real effect")

    # Plot battery over time
    fig, ax = plt.subplots(figsize=(10, 5))
    times = np.arange(len(battery_log)) * DT
    ax.plot(times, np.array(battery_log) * 100, 'g-', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Battery (%)')
    ax.set_title('Battery drain across a complete naive flight')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig('outputs/diagnostic_battery.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved outputs/diagnostic_battery.png")


# ======================================================================
# 4. Coordinate transform consistency
# ======================================================================
def check_coordinates():
    print("\n" + "=" * 60)
    print("CHECK 4: Coordinate transform round-trip")
    print("=" * 60)

    env = AORVAEnv()
    env.reset(seed=42)
    vg = env.voxel_grid

    test_points = [
        ("Westmead",  -33.8078, 150.9875, 100.0),
        ("Liverpool", -33.9173, 150.9233, 100.0),
        ("Mid-route", -33.8625, 150.9554, 100.0),
    ]

    print(f"  {'Location':<12}  {'lat':>9}  {'lon':>9}  {'world (x,y,z)':>22}  "
          f"{'voxel (i,j,k)':>16}")

    all_passed = True
    for name, lat, lon, alt in test_points:
        world = env._latlon_to_world(lat, lon, alt)
        voxel = env._world_to_voxel(world)

        # Round-trip back: voxel -> grid_to_latlon
        rt_lat, rt_lon, rt_alt = vg.grid_to_latlon(*voxel)

        print(f"  {name:<12}  {lat:>9.4f}  {lon:>9.4f}  "
              f"({world[0]:>6.0f},{world[1]:>6.0f},{world[2]:>4.0f})  "
              f"({voxel[0]:>4d},{voxel[1]:>4d},{voxel[2]:>4d})")

        lat_err = abs(rt_lat - lat)
        lon_err = abs(rt_lon - lon)
        if lat_err > 0.001 or lon_err > 0.001:
            print(f"      FAIL  Round-trip error: "
                  f"d_lat={lat_err:.5f}, d_lon={lon_err:.5f}")
            all_passed = False

    if all_passed:
        print("  PASS  All round-trips within 0.001 deg tolerance "
              "(~100m, expected for voxel snap)")


# ======================================================================
# Main
# ======================================================================
def main():
    import os
    os.makedirs('outputs', exist_ok=True)

    print("\n" + "#" * 60)
    print("# AORVA Physics Diagnostics")
    print("#" * 60)

    check_velocity_response()
    check_wind_magnitudes()
    check_energy_budget()
    check_coordinates()

    print("\n" + "#" * 60)
    print("# Diagnostics complete. Review plots in outputs/")
    print("#" * 60)


if __name__ == "__main__":
    main()
