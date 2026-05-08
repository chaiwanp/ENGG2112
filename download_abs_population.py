"""
download_abs_population.py

Downloads ABS 2021 Census population data for SA2 regions within the
Westmead-Liverpool corridor, computes population density (people/km^2),
and rasterizes it into a 2-D array that VoxelGrid3D stores as density_map.

Data sources (both free, no API key required)
---------------------------------------------
  SA2 boundaries : ABS ASGS2021 ArcGIS REST service
                   https://geo.abs.gov.au/arcgis/rest/services/ASGS2021/SA2/
  Population     : ABS Regional Statistics SDMX API
                   https://api.data.abs.gov.au/data/ABS,ABS_REGIONAL_SA2,1.0.0/

Risk weight function  W(rho) = tanh(rho / rho_ref)
---------------------------------------------
  rho      = population density in people/km^2
  rho_ref  = DENSITY_REFERENCE = 5 000 people/km^2

Justification (CASA BVLOS / JARUS SORA alignment)
  - Monotonically increasing - more people -> higher ground risk.
  - Bounded [0, 1] - directly usable as a reward coefficient.
  - Sub-linear (tanh) - mirrors diminishing-returns behaviour in
    JARUS SORA GRC classes (GRC 2->3 jump is larger than GRC 5->6).
  - Smooth and differentiable - produces clean gradient signal for RL.
  - Reference points:
      rho =    500 ppl/km^2  ->  W ~ 0.10  (outer suburban)
      rho =  2 500 ppl/km^2  ->  W ~ 0.46  (typical Westmead/Liverpool)
      rho =  5 000 ppl/km^2  ->  W ~ 0.76  (dense suburban - JARUS GRC 5)
      rho = 10 000 ppl/km^2  ->  W ~ 0.96  (inner-city - JARUS GRC 6)

Public API
----------
  population_risk_weight(density)         - pure function, no I/O
  download_population_density(clip_bounds) - download + join + density calc
  rasterize_population(density_gdf, voxel_grid) - GeoDataFrame -> 2-D array
"""

import io
import json
import os
import tempfile
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import requests


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Study-area bounding box (min_lon, min_lat, max_lon, max_lat)
WEST, SOUTH, EAST, NORTH = 150.8233, -34.0173, 151.0875, -33.7078

# ABS ArcGIS REST - SA2 Statistical Area Level 2 boundaries (2021 ASGS)
_SA2_BOUNDARY_URL = (
    "https://geo.abs.gov.au/arcgis/rest/"
    "services/ASGS2021/SA2/MapServer/0/query"
)

# Primary: ABS 2021 Census DataPack (static ZIP - real census data, always available)
# Contains Table G01: Selected Person Characteristics by SA2 for NSW
_DATAPACK_URL = (
    "https://www.abs.gov.au/census/find-census-data/datapacks/download/"
    "2021_GCP_SA2_for_NSW_short-header.zip"
)

# Secondary fallbacks: ABS SDMX API variants
_SDMX_URLS = [
    ("https://api.data.abs.gov.au/data/"
     "ABS,ABS_REGIONAL_SA2,1.0.0/ERP..A"
     "?detail=dataonly&format=jsondata&startPeriod=2021&endPeriod=2021"),
    ("https://api.data.abs.gov.au/data/"
     "ABS_REGIONAL_SA2/ERP..A"
     "?detail=dataonly&format=jsondata&startPeriod=2021&endPeriod=2021"),
]

# Risk weight reference density (people/km^2)
DENSITY_REFERENCE = 5_000.0


# ---------------------------------------------------------------------------
# Risk weight function
# ---------------------------------------------------------------------------

def population_risk_weight(density_people_per_km2):
    """
    Map population density -> ground-risk weight in [0, 1].

    Uses W(rho) = tanh(rho / rho_ref) where rho_ref = 5 000 ppl/km^2.

    Parameters
    ----------
    density_people_per_km2 : float or array-like

    Returns
    -------
    float or np.ndarray in [0, 1]
    """
    rho = np.asarray(density_people_per_km2, dtype=float)
    return np.tanh(rho / DENSITY_REFERENCE)


# ---------------------------------------------------------------------------
# Data download helpers
# ---------------------------------------------------------------------------

def _fetch_sa2_boundaries(west=WEST, south=SOUTH, east=EAST, north=NORTH):
    """
    Fetch SA2 polygon boundaries from the ABS ArcGIS REST service,
    filtered to the study-area bounding box.

    Returns
    -------
    GeoDataFrame (EPSG:4326) with columns:
        geometry, SA2_CODE_2021, SA2_NAME_2021, AREA_ALBERS_SQKM
    """
    params = {
        'where':             '1=1',
        'geometry':          f'{west},{south},{east},{north}',
        'geometryType':      'esriGeometryEnvelope',
        'inSR':              '4326',   # tells server bbox coords are WGS84
        'spatialRel':        'esriSpatialRelIntersects',
        'outFields':         '*',      # request all fields; parse flexibly below
        'outSR':             '4326',
        'returnGeometry':    'true',
        'f':                 'geojson',
        'resultRecordCount': 2000,
    }
    print("Fetching SA2 boundaries from ABS ArcGIS REST service...")
    resp = requests.get(_SA2_BOUNDARY_URL, params=params, timeout=60)
    resp.raise_for_status()

    with tempfile.NamedTemporaryFile(
            mode='w', suffix='.geojson', delete=False, encoding='utf-8') as tmp:
        json.dump(resp.json(), tmp)
        tmp_path = tmp.name

    try:
        gdf = gpd.read_file(tmp_path)
    finally:
        os.unlink(tmp_path)

    print(f"  Retrieved {len(gdf)} SA2 regions")
    if len(gdf) == 0:
        raise RuntimeError(
            "ABS ArcGIS returned 0 SA2 regions. "
            "Service may be down - try again later."
        )

    # Normalise field names: ABS sometimes uses SA2_CODE21 vs SA2_CODE_2021 etc.
    col_map = {}
    for col in gdf.columns:
        u = col.upper()
        if 'SA2' in u and 'CODE' in u and 'SA2_CODE_2021' not in gdf.columns:
            col_map[col] = 'SA2_CODE_2021'
        elif 'SA2' in u and 'NAME' in u and 'SA2_NAME_2021' not in gdf.columns:
            col_map[col] = 'SA2_NAME_2021'
        elif 'AREA' in u and 'KM' in u and 'AREA_ALBERS_SQKM' not in gdf.columns:
            col_map[col] = 'AREA_ALBERS_SQKM'
    if col_map:
        gdf = gdf.rename(columns=col_map)

    # Ensure required columns exist
    for col, default in [('SA2_CODE_2021', '000000000'),
                          ('SA2_NAME_2021', 'Unknown'),
                          ('AREA_ALBERS_SQKM', None)]:
        if col not in gdf.columns:
            gdf[col] = default

    return gdf


def _fetch_population_datapack():
    """
    Download the ABS 2021 Census DataPack for NSW (SA2 level) and extract
    usual resident population from Table G01.

    Returns
    -------
    dict mapping SA2_CODE_2021 (str) -> population (int)
    """
    print("  Downloading ABS 2021 Census DataPack (NSW SA2, ~25 MB)...")
    resp = requests.get(
        _DATAPACK_URL, timeout=180,
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        # Table G01 = Selected Person Characteristics (includes total pop)
        g01_name = next(
            (n for n in zf.namelist()
             if 'G01' in n.upper() and n.endswith('.csv')),
            None
        )
        if g01_name is None:
            raise RuntimeError(
                f"G01 table not found in DataPack zip. "
                f"Files found: {zf.namelist()[:10]}"
            )
        with zf.open(g01_name) as f:
            df = pd.read_csv(f)

    # Locate SA2 code column (short-header uses abbreviated names)
    code_col = next(
        (c for c in df.columns if 'SA2' in c.upper() and 'CODE' in c.upper()),
        None
    )
    # Locate total persons column (Tot_P_P in short-header GCP)
    pop_col = next(
        (c for c in df.columns
         if c in ('Tot_P_P', 'P_Tot_Tot', 'Tot_P_Tot', 'Total_P')),
        None
    )
    if not code_col or not pop_col:
        raise RuntimeError(
            f"Could not identify SA2 code or population column.\n"
            f"Available columns: {list(df.columns[:20])}"
        )

    pop_map = {
        str(code): int(pop)
        for code, pop in zip(df[code_col], df[pop_col])
        if pd.notna(pop) and int(pop) > 0
    }
    print(f"  DataPack: {len(pop_map)} SA2 populations loaded from {g01_name}")
    return pop_map


def _parse_sdmx_response(data):
    """Extract SA2_code -> population from an ABS SDMX-JSON response."""
    dims   = data['data']['structure']['dimensions']['series']
    series = data['data']['dataSets'][0]['series']

    geo_dim_idx = None
    for i, dim in enumerate(dims):
        if dim.get('id', '').upper() in ('REGION', 'SA2', 'ASGS_2021',
                                          'SA2_CODE', 'SA2CODE'):
            geo_dim_idx = i
            break
    if geo_dim_idx is None:
        geo_dim_idx = max(range(len(dims)), key=lambda i: len(dims[i]['values']))

    geo_values = dims[geo_dim_idx]['values']
    pop_by_sa2 = {}
    for key_str, obs_data in series.items():
        parts   = key_str.split(':')
        geo_idx = int(parts[geo_dim_idx])
        sa2_id  = str(geo_values[geo_idx]['id'])
        obs     = obs_data.get('observations', {})
        if obs:
            raw = list(obs.values())[0][0]
            if raw is not None:
                pop_by_sa2[sa2_id] = int(raw)
    return pop_by_sa2


def _fetch_sa2_population():
    """
    Fetch 2021 Census population by SA2.

    Tries in order:
      1. ABS Census DataPack ZIP (real 2021 census data, most reliable)
      2. ABS SDMX API fallbacks
    Raises RuntimeError if all sources fail - no fake data.

    Returns
    -------
    dict mapping SA2_CODE_2021 (str) -> population (int)
    """
    print("Fetching 2021 SA2 population...")

    # 1. Census DataPack (primary - static file, always available)
    try:
        return _fetch_population_datapack()
    except Exception as exc:
        print(f"  DataPack failed: {exc}")

    # 2. SDMX API fallbacks
    for url in _SDMX_URLS:
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            pop_map = _parse_sdmx_response(resp.json())
            print(f"  SDMX: {len(pop_map)} SA2 populations loaded")
            return pop_map
        except Exception as exc:
            print(f"  SDMX endpoint failed: {exc}")

    raise RuntimeError(
        "All population data sources failed.\n"
        "Check your internet connection and try again.\n"
        "DataPack URL: " + _DATAPACK_URL
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_population_density(
        clip_bounds=(WEST, SOUTH, EAST, NORTH)):
    """
    Download SA2 boundaries and 2021 ERP, compute population density
    and risk weight for each SA2 in the study area.

    Parameters
    ----------
    clip_bounds : (west, south, east, north)

    Returns
    -------
    GeoDataFrame with columns:
        geometry, SA2_CODE_2021, SA2_NAME_2021,
        population, area_km2, density_ppl_km2, risk_weight
    """
    gdf     = _fetch_sa2_boundaries(*clip_bounds)
    pop_map = _fetch_sa2_population()

    gdf['SA2_CODE_2021'] = gdf['SA2_CODE_2021'].astype(str)

    gdf['population'] = (gdf['SA2_CODE_2021']
                         .map(pop_map)
                         .fillna(0)
                         .astype(int))
    matched = int((gdf['population'] > 0).sum())
    print(f"  Matched population for {matched}/{len(gdf)} SA2 regions")

    # Area: prefer the ABS Albers area column; compute from geometry as fallback
    if 'AREA_ALBERS_SQKM' in gdf.columns:
        gdf['area_km2'] = pd.to_numeric(
            gdf['AREA_ALBERS_SQKM'], errors='coerce').fillna(0.01)
    else:
        proj = gdf.to_crs('EPSG:3577')        # GDA2020 Albers (metres)
        gdf['area_km2'] = proj.geometry.area / 1e6

    gdf['area_km2']        = gdf['area_km2'].clip(lower=0.01)
    gdf['density_ppl_km2'] = gdf['population'] / gdf['area_km2']
    gdf['risk_weight']     = population_risk_weight(gdf['density_ppl_km2'])

    keep = ['geometry', 'SA2_CODE_2021', 'SA2_NAME_2021',
            'population', 'area_km2', 'density_ppl_km2', 'risk_weight']
    result = gdf[keep].copy()

    print(f"\nDensity range: "
          f"{result['density_ppl_km2'].min():.0f}-"
          f"{result['density_ppl_km2'].max():.0f} ppl/km^2")
    print(f"Risk weight range: "
          f"{result['risk_weight'].min():.3f}-"
          f"{result['risk_weight'].max():.3f}")
    return result


def rasterize_population(density_gdf, voxel_grid):
    """
    Rasterize SA2 population density into a 2-D grid aligned with voxel_grid.

    Each grid column (ix, iy) is assigned the density of the SA2 whose
    polygon contains the column's lat/lon centre. Uses a vectorized
    geopandas spatial join for efficiency.

    Parameters
    ----------
    density_gdf : GeoDataFrame  - output of download_population_density()
    voxel_grid  : VoxelGrid3D

    Returns
    -------
    np.ndarray, shape (nx, ny), dtype float32, values in people/km^2
    """
    print(f"Rasterizing population into "
          f"{voxel_grid.nx} x {voxel_grid.ny} grid columns...")

    # --- Build a GeoDataFrame of all grid-column centres ---
    min_lon, min_lat, _, _ = voxel_grid.bounds
    vs = voxel_grid.voxel_size_m

    ix_all = np.arange(voxel_grid.nx)
    iy_all = np.arange(voxel_grid.ny)
    IX, IY = np.meshgrid(ix_all, iy_all, indexing='ij')   # shape (nx, ny)

    # Voxel-centre coordinates (+0.5 shifts to cell centre)
    lons = (min_lon + (IX + 0.5) * vs / voxel_grid.m_per_deg_lon).ravel()
    lats = (min_lat + (IY + 0.5) * vs / voxel_grid.m_per_deg_lat).ravel()

    grid_pts = gpd.GeoDataFrame(
        {'ix': IX.ravel().astype(np.int32),
         'iy': IY.ravel().astype(np.int32)},
        geometry=gpd.points_from_xy(lons, lats),
        crs='EPSG:4326',
    )

    # --- Spatial join: assign SA2 density to each grid column ---
    joined = gpd.sjoin(
        grid_pts,
        density_gdf[['geometry', 'density_ppl_km2']].reset_index(drop=True),
        how='left',
        predicate='within',
    )

    # Handle duplicate matches (edge cases on polygon boundaries)
    joined = joined.groupby(['ix', 'iy'], as_index=False)['density_ppl_km2'].max()

    # --- Fill output array ---
    density_map = np.zeros((voxel_grid.nx, voxel_grid.ny), dtype=np.float32)
    valid = joined.dropna(subset=['density_ppl_km2'])
    density_map[valid['ix'].values, valid['iy'].values] = (
        valid['density_ppl_km2'].values.astype(np.float32)
    )

    coverage = 100.0 * np.sum(density_map > 0) / density_map.size
    print(f"  Coverage: {coverage:.1f}% of grid columns assigned")
    print(f"  Density range: "
          f"{density_map.min():.0f}-{density_map.max():.0f} ppl/km^2")
    return density_map


# ======================================================================
# Download + rasterize script  (python download_abs_population.py)
# ======================================================================
if __name__ == "__main__":
    import pickle
    import matplotlib.pyplot as plt
    from voxel_grid_builder import VoxelGrid3D

    os.makedirs('data',    exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    # --- Download and save density GeoDataFrame ---
    density_gdf = download_population_density()
    density_gdf.to_file('data/population_density_sa2.gpkg', driver='GPKG')
    print("Saved data/population_density_sa2.gpkg")

    # --- Rasterize into voxel grid ---
    voxel_grid  = VoxelGrid3D.load('data/voxel_grid_westmead_liverpool.pkl')
    density_map = rasterize_population(density_gdf, voxel_grid)

    voxel_grid.add_population_density(density_map)
    voxel_grid.save('data/voxel_grid_westmead_liverpool.pkl')
    print("Updated voxel grid with population density layer.")

    # --- Visualise ---
    risk_map = population_risk_weight(density_map)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Hospital grid indices
    wx, wy, _ = voxel_grid.latlon_to_grid(-33.8078, 150.9875, 0)
    lx, ly, _ = voxel_grid.latlon_to_grid(-33.9173, 150.9233, 0)

    # Panel 1: population density heatmap
    im0 = axes[0].imshow(density_map.T, origin='lower', cmap='YlOrRd',
                          vmin=0,
                          extent=[0, voxel_grid.nx, 0, voxel_grid.ny],
                          aspect='auto')
    cb0 = plt.colorbar(im0, ax=axes[0], fraction=0.03, pad=0.02)
    cb0.set_label('Population density (people/km^2)', fontsize=9)
    axes[0].plot(wx, wy, 'b*', markersize=20, markeredgecolor='white',
                 markeredgewidth=1.5, label='Westmead Hospital', zorder=6)
    axes[0].plot(lx, ly, 'g*', markersize=20, markeredgecolor='white',
                 markeredgewidth=1.5, label='Liverpool Hospital', zorder=6)
    axes[0].plot([wx, lx], [wy, ly], 'k--', linewidth=1.2,
                 alpha=0.5, label='Corridor', zorder=5)
    axes[0].set_title(
        f'ABS 2021 Census - Population Density\n'
        f'max {density_map.max():.0f} ppl/km^2  |  '
        f'{voxel_grid.nx}x{voxel_grid.ny} grid, 50 m voxels',
        fontsize=10,
    )
    axes[0].set_xlabel('Grid X  (west -> east)', fontsize=9)
    axes[0].set_ylabel('Grid Y  (south -> north)', fontsize=9)
    axes[0].legend(loc='upper right', fontsize=8, framealpha=0.9)
    axes[0].grid(True, alpha=0.2, linestyle='--')

    # Panel 2: JARUS-aligned risk weight heatmap
    im1 = axes[1].imshow(risk_map.T, origin='lower', cmap='RdYlGn_r',
                          vmin=0, vmax=1,
                          extent=[0, voxel_grid.nx, 0, voxel_grid.ny],
                          aspect='auto')
    cb1 = plt.colorbar(im1, ax=axes[1], fraction=0.03, pad=0.02)
    cb1.set_label('Ground risk weight  W in [0, 1]', fontsize=9)

    # JARUS GRC reference lines on colorbar
    for rho_ref, label in [(500, 'GRC 3\n500'), (2500, 'GRC 4\n2500'),
                            (5000, 'GRC 5\n5000'), (10000, 'GRC 6\n10k')]:
        w = float(np.tanh(rho_ref / 5000.0))
        cb1.ax.axhline(w, color='black', linewidth=0.8, alpha=0.6)
        cb1.ax.text(1.08, w, label, fontsize=6, va='center',
                    transform=cb1.ax.transData)

    axes[1].plot(wx, wy, 'b*', markersize=20, markeredgecolor='white',
                 markeredgewidth=1.5, label='Westmead Hospital', zorder=6)
    axes[1].plot(lx, ly, 'g*', markersize=20, markeredgecolor='white',
                 markeredgewidth=1.5, label='Liverpool Hospital', zorder=6)
    axes[1].plot([wx, lx], [wy, ly], 'k--', linewidth=1.2,
                 alpha=0.5, label='Corridor', zorder=5)
    axes[1].set_title(
        'JARUS-Aligned Ground Risk  W(rho) = tanh(rho / 5000)\n'
        'GRC 3-6 reference lines shown on colorbar',
        fontsize=10,
    )
    axes[1].set_xlabel('Grid X  (west -> east)', fontsize=9)
    axes[1].set_ylabel('Grid Y  (south -> north)', fontsize=9)
    axes[1].legend(loc='upper right', fontsize=8, framealpha=0.9)
    axes[1].grid(True, alpha=0.2, linestyle='--')

    plt.suptitle('Phase 1 - ABS 2021 Census Population Risk Layer\n'
                 '(Used by AORVAEnv reward function, w2 term)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/population_risk_map.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved outputs/population_risk_map.png")

    # --- Summary stats ---
    print("\n=== SA2 Density Statistics ===")
    print(density_gdf[['SA2_NAME_2021', 'population',
                        'area_km2', 'density_ppl_km2',
                        'risk_weight']].sort_values(
        'density_ppl_km2', ascending=False).to_string(index=False))
