# Required packages:
"""
pip install osmnx geopandas pandas numpy matplotlib folium requests
pip install pyproj shapely rtree
"""


import osmnx as ox
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, box
import matplotlib.pyplot as plt
from gymnasium import spaces

# Define area of interest: Westmead to Liverpool
westmead_coords = (-33.8078, 150.9875)  # Westmead Hospital
liverpool_coords = (-33.9173, 150.9233)  # Liverpool Hospital

# Create bounding box with buffer (5km each side)
buffer_km = 5.0
north = max(westmead_coords[0], liverpool_coords[0]) + buffer_km/111
south = min(westmead_coords[0], liverpool_coords[0]) - buffer_km/111
east = max(westmead_coords[1], liverpool_coords[1]) + buffer_km/111
west = min(westmead_coords[1], liverpool_coords[1]) - buffer_km/111

bbox = (north, south, east, west)

print(f"Bounding Box: North={north:.4f}, South={south:.4f}, East={east:.4f}, West={west:.4f}")