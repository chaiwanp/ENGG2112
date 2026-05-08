"""
install.py — AORVA dependency installer.

Run once to install all required packages:
    python install.py

This is equivalent to:
    pip install -r requirements.txt
"""

import subprocess
import sys


def main():
    print("Installing AORVA dependencies from requirements.txt...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        check=False,
    )
    if result.returncode == 0:
        print("\nAll dependencies installed successfully.")
        print("\nPipeline execution order:")
        print("  python scripts/00_download_real_wind.py   # real ERA5 wind data (6x7 nodes)")
        print("  python scripts/01_download_buildings.py   # OSM buildings")
        print("  python scripts/02_build_voxel_grid.py    # 3D voxel occupancy grid")
        print("  python scripts/02b_download_population.py # ABS 2021 population layer")
        print("  python scripts/03_plan_trajectory.py     # A* reference path + checkpoints")
        print("  python scripts/04_train_agents.py ppo    # train PPO agent")
        print("  python scripts/04_train_agents.py sac    # train SAC agent")
        print("  python scripts/05_visualize.py           # visualise grid + wind field")
        print("  python scripts/06_evaluate.py            # evaluate trained agents")
    else:
        print(f"\nInstallation failed (exit code {result.returncode}).")
        print("Try manually: pip install -r requirements.txt")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
