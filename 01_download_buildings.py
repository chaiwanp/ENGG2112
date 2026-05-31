"""
Step 1 — Download building footprints from OpenStreetMap.

Output: data/buildings_westmead_liverpool.gpkg  (or .geojson / .pkl fallback)
        outputs/buildings_map.png

Usage:
    python scripts/01_download_buildings.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from download_buildings import download_buildings, save_buildings, visualize_buildings

if __name__ == "__main__":
    buildings = download_buildings()
    save_buildings(buildings)
    visualize_buildings(buildings)
    print("\nStep 1 complete.")
