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

## Pipeline — run in order

### Step 0 — Download real wind data
```bash
python scripts/00_download_real_wind.py
# Spatial grid (recommended, enables IDW):
python scripts/00_download_real_wind.py --spatial --days 30

# Outputs: data/wind_spatial_real.csv
```

### Step 1 — Download buildings
```bash
python scripts/01_download_buildings.py
# Outputs: data/buildings_westmead_liverpool.gpkg
#          outputs/buildings_map.png
```

### Step 2 — Build voxel grid
```bash
python scripts/02_build_voxel_grid.py
# Outputs: data/voxel_grid_westmead_liverpool.pkl
```

### Step 2b — Add population density layer (optional)
```bash
python scripts/02b_download_population.py
# Updates: data/voxel_grid_westmead_liverpool.pkl
# Outputs: data/population_density_sa2.gpkg
#          outputs/population_risk_map.png
```

### Step 3 — Plan A* reference trajectory
```bash
python scripts/03_plan_trajectory.py
# Outputs: data/reference_trajectory.pkl
#          outputs/reference_trajectory.png
```

### Step 4 — Train RL agents
```bash
python scripts/04_train_agents.py both
python scripts/04_train_agents.py ppo --ppo-steps 500000
python scripts/04_train_agents.py sac --sac-steps 300000

# Monitor: tensorboard --logdir logs
# Outputs: models/ppo_aorva_final.zip
#          models/sac_aorva_final.zip
```

### Step 5 — Visualise
```bash
python scripts/05_visualise.py
# Outputs: outputs/voxel_slice_*.png
#          outputs/wind_field_*.png
```

### Step 6 — Evaluate (Phase 4 metrics)
```bash
python scripts/06_evaluate.py
python scripts/06_evaluate.py --episodes 50   # faster smoke test

# Outputs: outputs/eval_training_curves.png
#          outputs/eval_checkpoint_timing.png
#          outputs/eval_trajectory_map.png
#          outputs/eval_comparison_table.txt  ← delta_sigma result
```

---

## Key metrics (Phase 4)

| Metric | Description |
|--------|-------------|
| RMSE deviation (s) | Root-mean-square of arrival-time error across all 15 checkpoints |
| Safety violation rate | % of episodes with altitude / NFZ / collision events |
| Δσ | σ_ambulance − σ_drone — headline reliability result |

Ambulance baseline: Westmead → Liverpool, mean ~35 min, σ ~12 min  
(estimated from TomTom traffic data; replace with measured data for report).

---

## Wind data

The default pipeline uses **ERA5 reanalysis** data via the Open-Meteo API  
(free, no API key, same dataset as BoM reanalysis products).

The spatial-grid download (`--spatial`) fetches 42 nodes over the study area  
and enables **Inverse Distance Weighting (IDW)** inside `WindField3D`, so  
each position in the voxel grid gets its own spatially-interpolated wind vector.

---

## Install

```bash
pip install gymnasium stable-baselines3 osmnx geopandas scipy \
            pandas numpy matplotlib requests
```
