"""
download_buildings.py

Downloads building footprints from OpenStreetMap for the
Westmead–Liverpool corridor and saves them to data/.

Exposes three importable functions used by scripts/01_download_buildings.py:
    download_buildings()    -> GeoDataFrame
    save_buildings(gdf)
    visualize_buildings(gdf)

Can also be run directly:
    python download_buildings.py
"""

from __future__ import annotations

import os
import pickle
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

_DIR        = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR   = os.path.join(_DIR, "data")
_OUTPUT_DIR = os.path.join(_DIR, "outputs")

# Westmead–Liverpool bounding box
NORTH, SOUTH, EAST, WEST = -33.7078, -34.0173, 151.0875, 150.8233

WESTMEAD_COORDS  = (150.9875, -33.8078)   # (lon, lat)
LIVERPOOL_COORDS = (150.9233, -33.9173)


# ======================================================================
# Height extraction
# ======================================================================

def _extract_height(row) -> float:
    """Extract building height from OSM tags, falling back to type defaults."""
    if "height" in row and pd.notna(row["height"]):
        try:
            return float(str(row["height"]).replace("m", "").replace("M", "").strip())
        except ValueError:
            pass

    if "building:levels" in row and pd.notna(row["building:levels"]):
        try:
            return float(row["building:levels"]) * 3.5
        except ValueError:
            pass

    defaults = {
        "hospital": 25, "commercial": 15, "retail": 10,
        "residential": 12, "house": 6, "apartments": 20,
        "industrial": 10, "yes": 8,
    }
    return defaults.get(str(row.get("building", "yes")), 8)


# ======================================================================
# Core functions
# ======================================================================

def download_buildings() -> gpd.GeoDataFrame:
    """
    Download building polygons for the study area from OpenStreetMap.

    Returns
    -------
    GeoDataFrame with columns: geometry, height_m, building_type,
    building_name (where available), building_id.
    """
    import osmnx as ox

    os.makedirs(_DATA_DIR,   exist_ok=True)
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    print("Downloading buildings from OpenStreetMap...")
    print(f"  Bounds: N={NORTH}, S={SOUTH}, E={EAST}, W={WEST}")

    buildings = None
    for attempt, method in enumerate(["features_from_bbox",
                                       "geometries_from_bbox",
                                       "features_from_place"]):
        try:
            if method == "features_from_bbox":
                buildings = ox.features_from_bbox(
                    bbox=(NORTH, SOUTH, EAST, WEST),
                    tags={"building": True},
                )
            elif method == "geometries_from_bbox":
                buildings = ox.geometries_from_bbox(
                    north=NORTH, south=SOUTH, east=EAST, west=WEST,
                    tags={"building": True},
                )
            else:
                buildings = ox.features_from_place(
                    "Sydney, New South Wales, Australia",
                    tags={"building": True},
                )
                buildings = buildings.cx[WEST:EAST, SOUTH:NORTH]
            print(f"  Downloaded {len(buildings)} buildings (method: {method})")
            break
        except Exception as e:
            print(f"  {method} failed: {e}")

    if buildings is None or len(buildings) == 0:
        raise RuntimeError("Could not download buildings from OpenStreetMap.")

    # Keep polygons only
    buildings = buildings[
        buildings.geometry.type.isin(["Polygon", "MultiPolygon"])
    ].copy()
    print(f"  Filtered to {len(buildings)} building polygons")

    # Extract heights
    buildings["height_m"] = buildings.apply(_extract_height, axis=1)

    # Clean to essential columns
    buildings = buildings.reset_index(drop=False)
    keep = ["geometry", "height_m"]
    if "building" in buildings.columns:
        buildings["building_type"] = buildings["building"].astype(str)
        keep.append("building_type")
    if "name" in buildings.columns:
        buildings["building_name"] = buildings["name"].astype(str)
        keep.append("building_name")

    out = gpd.GeoDataFrame(
        buildings[keep], geometry="geometry", crs=buildings.crs
    )
    out["building_id"] = range(len(out))

    print(f"  Final dataset: {len(out)} buildings")
    return out


def save_buildings(buildings: gpd.GeoDataFrame) -> None:
    """Save to data/ in the best available format (GPKG → GeoJSON → pkl)."""
    os.makedirs(_DATA_DIR, exist_ok=True)
    base = os.path.join(_DATA_DIR, "buildings_westmead_liverpool")

    for path, driver in [(base + ".gpkg", "GPKG"),
                          (base + ".geojson", "GeoJSON")]:
        try:
            buildings.to_file(path, driver=driver)
            print(f"Saved buildings -> {path}")
            return
        except Exception as e:
            print(f"  {driver} failed: {e}")

    pkl_path = base + ".pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(buildings, f)
    print(f"Saved buildings -> {pkl_path}  (pickle fallback)")


def visualize_buildings(buildings: gpd.GeoDataFrame) -> None:
    """Save a building-height choropleth map to outputs/."""
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 12))
    buildings.plot(
        ax=ax, column="height_m", legend=True,
        cmap="YlOrRd", edgecolor="black", linewidth=0.1, alpha=0.7,
    )

    ax.plot(*WESTMEAD_COORDS,  "b*", markersize=25, label="Westmead Hospital",
            markeredgecolor="white", markeredgewidth=2, zorder=5)
    ax.plot(*LIVERPOOL_COORDS, "r*", markersize=25, label="Liverpool Hospital",
            markeredgecolor="white", markeredgewidth=2, zorder=5)
    ax.plot(
        [WESTMEAD_COORDS[0], LIVERPOOL_COORDS[0]],
        [WESTMEAD_COORDS[1], LIVERPOOL_COORDS[1]],
        "k--", linewidth=2, alpha=0.5, label="Flight Path", zorder=4,
    )

    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude",  fontsize=12)
    ax.set_title(
        "Building Heights: Westmead to Liverpool Corridor\n(Red = Taller)",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_aspect("equal")

    out = os.path.join(_OUTPUT_DIR, "buildings_map.png")
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved map -> {out}")

    print("\n=== Building height statistics ===")
    print(buildings["height_m"].describe().to_string())


# ======================================================================
# Direct execution
# ======================================================================
if __name__ == "__main__":
    buildings = download_buildings()
    save_buildings(buildings)
    visualize_buildings(buildings)
