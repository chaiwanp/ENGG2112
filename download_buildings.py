# download_buildings.py
import osmnx as ox
import geopandas as gpd
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# Create directories
os.makedirs('data', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Define bounding box (north, south, east, west)
north, south, east, west = -33.7078, -34.0173, 151.0875, 150.8233

# Download buildings with height data
tags = {'building': True}

print("Downloading buildings from OpenStreetMap...")
print(f"Bounding box: N={north:.4f}, S={south:.4f}, E={east:.4f}, W={west:.4f}")

try:
    # Correct syntax - pass bbox as positional argument
    buildings = ox.features_from_bbox(
        bbox=(north, south, east, west),
        tags=tags
    )
    print(f"Downloaded {len(buildings)} buildings")
    
except Exception as e:
    print(f"Error downloading buildings: {e}")
    print("\nTrying alternative method...")
    
    try:
        buildings = ox.geometries_from_bbox(
            north=north, south=south, east=east, west=west,
            tags=tags
        )
        print(f"Downloaded {len(buildings)} buildings using alternative method")
    except:
        print("Alternative method also failed. Trying features_from_place...")
        buildings = ox.features_from_place(
            "Sydney, New South Wales, Australia",
            tags=tags
        )
        print(f"Downloaded {len(buildings)} buildings from Sydney")
        
        # Filter to bounding box
        print("Filtering to study area...")
        buildings = buildings.cx[west:east, south:north]
        print(f"Filtered to {len(buildings)} buildings in study area")

# Filter to get polygons only
buildings_poly = buildings[buildings.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()
print(f"Filtered to {len(buildings_poly)} building polygons")

# Extract height information
def extract_height(row):
    """Extract building height from OSM tags"""
    # Check for height tag
    if 'height' in row and pd.notna(row['height']):
        try:
            height_str = str(row['height']).replace('m', '').replace('M', '').strip()
            return float(height_str)
        except:
            pass
    
    # Check for building:levels
    if 'building:levels' in row and pd.notna(row['building:levels']):
        try:
            levels = float(row['building:levels'])
            return levels * 3.5  # Assume 3.5m per floor
        except:
            pass
    
    # Default height based on building type
    building_type = row.get('building', 'yes')
    default_heights = {
        'hospital': 25,
        'commercial': 15,
        'retail': 10,
        'residential': 12,
        'house': 6,
        'apartments': 20,
        'industrial': 10,
        'yes': 8  # default
    }
    return default_heights.get(building_type, 8)

print("\nExtracting building heights...")
buildings_poly['height_m'] = buildings_poly.apply(extract_height, axis=1)

# ============================================
# CLEAN DATA BEFORE SAVING 
# ============================================
print("\nCleaning data for saving...")

# Reset index to avoid index issues
buildings_poly = buildings_poly.reset_index(drop=False)

# Keep only essential columns and geometry
essential_columns = ['geometry', 'height_m']

# Try to keep building type if available
if 'building' in buildings_poly.columns:
    essential_columns.append('building')
    buildings_poly['building_type'] = buildings_poly['building'].astype(str)
    essential_columns.append('building_type')

# Try to keep building name if available
if 'name' in buildings_poly.columns:
    buildings_poly['building_name'] = buildings_poly['name'].astype(str)
    essential_columns.append('building_name')

# Create clean dataframe with only essential columns
buildings_clean = gpd.GeoDataFrame(
    buildings_poly[essential_columns],
    geometry='geometry',
    crs=buildings_poly.crs
)

# Add unique ID
buildings_clean['building_id'] = range(len(buildings_clean))

print(f"Cleaned data: {len(buildings_clean)} buildings, {len(buildings_clean.columns)} columns")
print(f"Columns: {list(buildings_clean.columns)}")

# Save to file - try multiple formats
print("\nSaving buildings to file...")

# Try GPKG first
try:
    buildings_clean.to_file('data/buildings_westmead_liverpool.gpkg', driver='GPKG')
    print(f"[OK] Saved as GPKG: {len(buildings_clean)} buildings")
except Exception as e:
    print(f"[ERROR] GPKG save failed: {e}")
    
    # Fallback to GeoJSON
    try:
        buildings_clean.to_file('data/buildings_westmead_liverpool.geojson', driver='GeoJSON')
        print(f"[OK] Saved as GeoJSON: {len(buildings_clean)} buildings")
    except Exception as e2:
        print(f"[ERROR] GeoJSON save failed: {e2}")
        
        # Final fallback to pickle
        print("Using pickle format as final fallback...")
        with open('data/buildings_westmead_liverpool.pkl', 'wb') as f:
            pickle.dump(buildings_clean, f)
        print(f"[OK] Saved as pickle: {len(buildings_clean)} buildings")

# Quick visualization
print("\nCreating visualization...")
fig, ax = plt.subplots(figsize=(14, 12))

# Plot buildings
buildings_clean.plot(
    ax=ax, 
    column='height_m', 
    legend=True, 
    cmap='YlOrRd',
    edgecolor='black', 
    linewidth=0.1,
    alpha=0.7
)

# Mark hospitals
westmead_coords = (150.9875, -33.8078)  # lon, lat
liverpool_coords = (150.9233, -33.9173)

ax.plot(westmead_coords[0], westmead_coords[1], 'b*', markersize=25, 
        label='Westmead Hospital', markeredgecolor='white', markeredgewidth=2, zorder=5)
ax.plot(liverpool_coords[0], liverpool_coords[1], 'r*', markersize=25, 
        label='Liverpool Hospital', markeredgecolor='white', markeredgewidth=2, zorder=5)

# Draw line between hospitals
ax.plot([westmead_coords[0], liverpool_coords[0]], 
        [westmead_coords[1], liverpool_coords[1]], 
        'k--', linewidth=2, alpha=0.5, label='Flight Path', zorder=4)

ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_title('Building Heights: Westmead to Liverpool Corridor\n(Red = Taller Buildings)', 
             fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')

# Set aspect ratio to be equal
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('outputs/buildings_map.png', dpi=300, bbox_inches='tight')
print("[OK] Saved visualization to outputs/buildings_map.png")
plt.show()

# Statistics
print("\n" + "="*60)
print("BUILDING HEIGHT STATISTICS")
print("="*60)
print(buildings_clean['height_m'].describe())

print("\n" + "="*60)
print("SPATIAL EXTENT")
print("="*60)
bounds = buildings_clean.total_bounds
print(f"Min Longitude: {bounds[0]:.4f}")
print(f"Min Latitude:  {bounds[1]:.4f}")
print(f"Max Longitude: {bounds[2]:.4f}")
print(f"Max Latitude:  {bounds[3]:.4f}")
print(f"Width:  {(bounds[2]-bounds[0])*111.32*np.cos(np.radians((bounds[1]+bounds[3])/2)):.2f} km")
print(f"Height: {(bounds[3]-bounds[1])*111.32:.2f} km")

if 'building_type' in buildings_clean.columns:
    print("\n" + "="*60)
    print("BUILDING TYPE DISTRIBUTION (Top 10)")
    print("="*60)
    print(buildings_clean['building_type'].value_counts().head(10))

print("\n" + "="*60)
print("DOWNLOAD COMPLETE!")
print("="*60)
print(f"Total buildings: {len(buildings_clean)}")
print(f"Data files created:")
if os.path.exists('data/buildings_westmead_liverpool.gpkg'):
    print(f"  [OK] data/buildings_westmead_liverpool.gpkg")
elif os.path.exists('data/buildings_westmead_liverpool.geojson'):
    print(f"  [OK] data/buildings_westmead_liverpool.geojson")
else:
    print(f"  [OK] data/buildings_westmead_liverpool.pkl")
print(f"  [OK] outputs/buildings_map.png")
print("="*60)