# AORVA — Autonomous Organ Routing via Autonomous Drones

ENGG2112 Multidisciplinary Engineering Project  
University of Sydney

---

## Repo rules
- **Don't commit to `main` if code is still being debugged.** Use a feature branch.  
- **Report assets (figures, CSVs for the report) go in a separate branch** — don't clutter `main` with them.  
- Prefix work-in-progress files with `#` in their name so everyone knows not to rely on them.

---

## Directory structure

```
AORVA/
│
├── scripts/                  ← Run these in order (00 → 06)
│   ├── 00_download_real_wind.py
│   ├── 01_download_buildings.py
│   ├── 02_build_voxel_grid.py
│   ├── 02b_download_population.py   (optional, adds ground-risk layer)
│   ├── 03_plan_trajectory.py
│   ├── 04_train_agents.py
│   ├── 05_visualise.py
│   └── 06_evaluate.py
│
├── data/                     ← Generated data files (gitignored)
├── models/                   ← Trained model checkpoints (gitignored)
├── logs/                     ← TensorBoard logs (gitignored)
├── outputs/                  ← Plots and result PNGs (gitignored)
│
│  ── Core modules ──
├── aorva_env.py              ← Gymnasium RL environment
├── trajectory_planner.py     ← A* pathfinding + checkpoint generation
├── voxel_grid_builder.py     ← 3D voxel occupancy grid
├── wind_field_interpolator.py← Wind field (log-law + IDW spatial)
├── download_buildings.py     ← OSM building download
├── download_bom_weather.py   ← ERA5 wind download (OpenMeteo) + BoM fallback
├── download_abs_population.py← ABS census ground-risk layer
├── evaluate_agents.py        ← PPO/SAC training entry point
├── visualize_voxel_grid.py   ← Voxel slice plotting helper
└── README.md
```

---
#install reqs
```bash
pip install gymnasium stable-baselines3 osmnx geopandas scipy \
            pandas numpy matplotlib requests
```
