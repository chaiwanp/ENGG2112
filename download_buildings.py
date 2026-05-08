"""
download_buildings.py

Downloads building footprints from OpenStreetMap for the Westmead-Liverpool
drone corridor, infers building heights from OSM tags, and saves the result
as a GeoPackage (with GeoJSON and pickle fallbacks).

Functions
---------
  extract_height(row)           - infer height from OSM tags or building type
  download_buildings(...)       - fetch polygons from OSM, return GeoDataFrame
  save_buildings(gdf, out_dir)  - write to GPKG / GeoJSON / pickle
  visualize_buildings(gdf, ...) - save a choropleth map of building heights

Run directly to download and save:
    python download_buildings.py
"""

import os
import pickle
import warnings

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd

warnings.filterwarnings('ignore')

# Geographic bounds of the study area
NORTH, SOUTH, EAST, WEST = -33.7078, -34.0173, 151.0875, 150.8233

# Hospital coordinates (lon, lat)
WESTMEAD  = (150.9875, -33.8078)
LIVERPOOL = (150.9233, -33.9173)

# Default building heights by OSM type (metres)
_HEIGHT_DEFAULTS = {
    'hospital': 25, 'commercial': 15, 'retail': 10,
    'residential': 12, 'house': 6, 'apartments': 20,
    'industrial': 10, 'yes': 8,
}


def extract_height(row):
    """
    Infer building height from an OSM feature row.

    Priority: explicit 'height' tag -> 'building:levels' x 3.5 m -> type default.
    """
    if 'height' in row and pd.notna(row['height']):
        try:
            return float(str(row['height']).replace('m', '').replace('M', '').strip())
        except ValueError:
            pass
    if 'building:levels' in row and pd.notna(row['building:levels']):
        try:
            return float(row['building:levels']) * 3.5
        except ValueError:
            pass
    building_type = row.get('building', 'yes')
    return _HEIGHT_DEFAULTS.get(str(building_type), 8)


def download_buildings(north=NORTH, south=SOUTH, east=EAST, west=WEST):
    """
    Fetch building polygons from OpenStreetMap for the given bounding box.

    Supports osmnx 2.x (bbox tuple = west, south, east, north) and
    osmnx 1.x (keyword args) automatically.

    Returns
    -------
    GeoDataFrame with columns: geometry, height_m, [building_type, building_name]
    """
    from shapely.geometry import box as shapely_box

    tags = {'building': True}
    print(f"osmnx {ox.__version__}")
    print("Downloading buildings from OpenStreetMap...")
    print(f"Bounds: N={north:.4f}  S={south:.4f}  E={east:.4f}  W={west:.4f}")

    osmnx_major = int(ox.__version__.split('.')[0])
    buildings = None
    errors = []

    # Method 1: features_from_bbox with version-correct bbox order
    try:
        if osmnx_major >= 2:
            # osmnx 2.x: (left, bottom, right, top) = (west, south, east, north)
            buildings = ox.features_from_bbox(bbox=(west, south, east, north), tags=tags)
        else:
            buildings = ox.features_from_bbox(
                north=north, south=south, east=east, west=west, tags=tags
            )
        print(f"Downloaded {len(buildings)} features via ox.features_from_bbox")
    except Exception as exc:
        errors.append(f"features_from_bbox: {exc}")

    # Method 2: features_from_polygon - avoids any bbox-format ambiguity
    if buildings is None:
        try:
            bbox_poly = shapely_box(west, south, east, north)
            buildings = ox.features_from_polygon(bbox_poly, tags=tags)
            print(f"Downloaded {len(buildings)} features via ox.features_from_polygon")
        except Exception as exc:
            errors.append(f"features_from_polygon: {exc}")

    # Method 3: geometries_from_bbox - osmnx 1.x only
    if buildings is None:
        try:
            buildings = ox.geometries_from_bbox(
                north=north, south=south, east=east, west=west, tags=tags
            )
            print(f"Downloaded {len(buildings)} features via ox.geometries_from_bbox")
        except Exception as exc:
            errors.append(f"geometries_from_bbox: {exc}")

    if buildings is None:
        err_detail = "\n".join(f"  {e}" for e in errors)
        raise RuntimeError(
            f"All OSM download methods failed:\n{err_detail}\n"
            "Check internet connection or try: pip install --upgrade osmnx"
        )

    # Keep only polygon geometries
    polys = buildings[
        buildings.geometry.type.isin(['Polygon', 'MultiPolygon'])
    ].copy()
    print(f"Kept {len(polys)} building polygons")

    print("Extracting building heights...")
    polys['height_m'] = polys.apply(extract_height, axis=1)

    # Build clean output GeoDataFrame with only essential columns
    polys = polys.reset_index(drop=False)
    keep  = ['geometry', 'height_m']
    if 'building' in polys.columns:
        polys['building_type'] = polys['building'].astype(str)
        keep.append('building_type')
    if 'name' in polys.columns:
        polys['building_name'] = polys['name'].astype(str)
        keep.append('building_name')

    clean = gpd.GeoDataFrame(polys[keep], geometry='geometry', crs=polys.crs)
    clean['building_id'] = range(len(clean))
    print(f"Final: {len(clean)} buildings, columns: {list(clean.columns)}")
    return clean


def save_buildings(buildings_gdf, out_dir='data'):
    """
    Persist the GeoDataFrame to disk.

    Tries GPKG -> GeoJSON -> pickle (in order). Returns the path written.
    """
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, 'buildings_westmead_liverpool')

    for path, driver in [(f'{stem}.gpkg', 'GPKG'), (f'{stem}.geojson', 'GeoJSON')]:
        try:
            buildings_gdf.to_file(path, driver=driver)
            print(f"Saved to {path}")
            return path
        except Exception as exc:
            print(f"{driver} save failed: {exc}")

    pkl_path = f'{stem}.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(buildings_gdf, f)
    print(f"Saved to {pkl_path} (pickle fallback)")
    return pkl_path


def visualize_buildings(buildings_gdf,
                        out_path='outputs/buildings_map.png'):
    """Save a choropleth map of building heights with statistics panel."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.lines import Line2D

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig = plt.figure(figsize=(18, 11))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[3, 1], figure=fig,
                            wspace=0.08)
    ax     = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])

    # Choropleth map
    buildings_gdf.plot(
        ax=ax, column='height_m', legend=True,
        cmap='YlOrRd', edgecolor='grey', linewidth=0.08, alpha=0.75,
        legend_kwds={'label': 'Building height (m)', 'fraction': 0.03,
                     'pad': 0.02},
    )

    # Hospital markers
    ax.plot(*WESTMEAD,  'b*', markersize=28, label='Westmead Hospital',
            markeredgecolor='white', markeredgewidth=2.0, zorder=6)
    ax.plot(*LIVERPOOL, 'g*', markersize=28, label='Liverpool Hospital',
            markeredgecolor='white', markeredgewidth=2.0, zorder=6)
    ax.plot([WESTMEAD[0], LIVERPOOL[0]],
            [WESTMEAD[1], LIVERPOOL[1]],
            'k--', linewidth=1.8, alpha=0.55, label='Corridor', zorder=5)

    ax.set_xlabel('Longitude', fontsize=9)
    ax.set_ylabel('Latitude',  fontsize=9)
    n = len(buildings_gdf)
    h_mean = buildings_gdf['height_m'].mean()
    ax.set_title(
        f'OpenStreetMap Buildings - Westmead-Liverpool Corridor\n'
        f'{n:,} buildings  |  mean height {h_mean:.1f} m  '
        f'|  voxel resolution 50 m',
        fontsize=11,
    )
    ax.legend(loc='upper right', framealpha=0.92, fontsize=9)
    ax.grid(True, alpha=0.25, linestyle='--')

    # Right panel: height distribution
    heights = buildings_gdf['height_m'].clip(upper=60)
    ax_bar.hist(heights, bins=30, orientation='horizontal',
                color='tomato', edgecolor='white', linewidth=0.4, alpha=0.85)
    ax_bar.axhline(h_mean, color='navy', linestyle='--', linewidth=1.5,
                   label=f'Mean {h_mean:.1f} m')
    ax_bar.set_xlabel('Count', fontsize=9)
    ax_bar.set_ylabel('Height (m)', fontsize=9)
    ax_bar.set_title('Height distribution\n(capped at 60 m)', fontsize=9)
    ax_bar.legend(fontsize=8)
    ax_bar.grid(True, alpha=0.3)

    plt.suptitle('Phase 1 - Building Data (OpenStreetMap)', fontsize=12,
                 fontweight='bold')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved map to {out_path}")


# ======================================================================
# Download script  (python download_buildings.py)
# ======================================================================
if __name__ == "__main__":
    buildings = download_buildings()
    save_buildings(buildings)
    visualize_buildings(buildings)

    print("\n=== Building Height Statistics ===")
    print(buildings['height_m'].describe())

    bounds = buildings.total_bounds
    width_km  = (bounds[2] - bounds[0]) * 111.32 * np.cos(
        np.radians((bounds[1] + bounds[3]) / 2))
    height_km = (bounds[3] - bounds[1]) * 111.32
    print(f"\nSpatial extent: {width_km:.2f} km x {height_km:.2f} km")

    if 'building_type' in buildings.columns:
        print("\nTop 10 building types:")
        print(buildings['building_type'].value_counts().head(10))

    print(f"\nTotal buildings saved: {len(buildings)}")
