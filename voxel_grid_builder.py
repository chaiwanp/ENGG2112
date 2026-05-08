"""
voxel_grid_builder.py

VoxelGrid3D: 3-D occupancy grid built from building-footprint GeoDataFrames.

Coordinate conventions
----------------------
  x  - east-west   (+ = eastward,  aligns with longitude)
  y  - north-south (+ = northward, aligns with latitude)
  z  - altitude    (+ = upward,    0 = ground level)

Each voxel is a cube of side `voxel_size_m` metres.
Grid values: 0 = free, 1 = occupied (see metadata for subtype).
Metadata:    0 = free, 1 = building, 2 = no-fly zone, 3 = high-risk ground.

Run directly to build the Westmead-Liverpool grid:
    python voxel_grid_builder.py
"""

import os
import pickle

import geopandas as gpd
import numpy as np


class VoxelGrid3D:
    """3-D occupancy grid aligned to a lat/lon bounding box."""

    def __init__(self, bounds, voxel_size_m=50, max_height_m=500):
        """
        Parameters
        ----------
        bounds : (min_lon, min_lat, max_lon, max_lat)
        voxel_size_m : float  - side length of each voxel in metres
        max_height_m : float  - ceiling altitude in metres
        """
        self.voxel_size_m = float(voxel_size_m)
        self.max_height_m = float(max_height_m)
        self.bounds = bounds

        min_lon, min_lat, max_lon, max_lat = bounds
        lat_centre = (min_lat + max_lat) / 2.0
        self.m_per_deg_lat = 111_320.0
        self.m_per_deg_lon = 111_320.0 * np.cos(np.radians(lat_centre))

        self.width_m  = (max_lon - min_lon) * self.m_per_deg_lon
        self.length_m = (max_lat - min_lat) * self.m_per_deg_lat

        self.nx = int(np.ceil(self.width_m  / self.voxel_size_m))
        self.ny = int(np.ceil(self.length_m / self.voxel_size_m))
        self.nz = int(np.ceil(self.max_height_m / self.voxel_size_m))

        print(f"Grid dimensions: {self.nx} x {self.ny} x {self.nz}")
        print(f"Total voxels: {self.nx * self.ny * self.nz:,}")

        self.grid     = np.zeros((self.nx, self.ny, self.nz), dtype=np.uint8)
        self.metadata = np.zeros((self.nx, self.ny, self.nz), dtype=np.uint8)

        # 2-D population density layer (people/km^2).
        # Populated by add_population_density(); zeros until then.
        self.density_map = np.zeros((self.nx, self.ny), dtype=np.float32)

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def latlon_to_grid(self, lat, lon, alt_m=0):
        """(lat, lon, alt_m) -> (ix, iy, iz), clamped to grid bounds."""
        min_lon, min_lat, _, _ = self.bounds
        x = int((lon - min_lon) * self.m_per_deg_lon / self.voxel_size_m)
        y = int((lat - min_lat) * self.m_per_deg_lat / self.voxel_size_m)
        z = int(alt_m / self.voxel_size_m)
        return (int(np.clip(x, 0, self.nx - 1)),
                int(np.clip(y, 0, self.ny - 1)),
                int(np.clip(z, 0, self.nz - 1)))

    def grid_to_latlon(self, x, y, z):
        """(ix, iy, iz) -> (lat, lon, alt_m) at voxel centre."""
        min_lon, min_lat, _, _ = self.bounds
        lon   = min_lon + (x * self.voxel_size_m) / self.m_per_deg_lon
        lat   = min_lat + (y * self.voxel_size_m) / self.m_per_deg_lat
        alt_m = z * self.voxel_size_m
        return lat, lon, alt_m

    # ------------------------------------------------------------------
    # Grid population
    # ------------------------------------------------------------------

    def add_buildings(self, buildings_gdf):
        """Rasterise building footprints into the occupancy grid."""
        print(f"Adding {len(buildings_gdf)} buildings to grid...")
        for idx, building in buildings_gdf.iterrows():
            geom   = building.geometry
            height = float(building['height_m'])
            minx, miny, maxx, maxy = geom.bounds
            x0, y0, _ = self.latlon_to_grid(miny, minx, 0)
            x1, y1, _ = self.latlon_to_grid(maxy, maxx, 0)
            z_max = int(height / self.voxel_size_m)
            for x in range(x0, min(x1 + 1, self.nx)):
                for y in range(y0, min(y1 + 1, self.ny)):
                    for z in range(0, min(z_max, self.nz)):
                        self.grid[x, y, z]     = 1
                        self.metadata[x, y, z] = 1
            if (idx + 1) % 1000 == 0:
                print(f"  processed {idx + 1} buildings...")
        occupied = int(np.sum(self.grid))
        total    = self.grid.size
        print(f"Occupied voxels: {occupied:,} ({100 * occupied / total:.2f}%)")

    def add_population_density(self, density_map_2d):
        """
        Store a 2-D population density layer in this grid.

        Parameters
        ----------
        density_map_2d : np.ndarray, shape (nx, ny), dtype float32
            Population density in people/km^2, produced by
            download_abs_population.rasterize_population().

        The layer is used by AORVAEnv to compute ground risk costs.
        Zero cells mean no data (treated as zero risk).
        """
        if density_map_2d.shape != (self.nx, self.ny):
            raise ValueError(
                f"density_map shape {density_map_2d.shape} does not match "
                f"grid (nx={self.nx}, ny={self.ny})"
            )
        self.density_map = density_map_2d.astype(np.float32)
        populated = int(np.sum(self.density_map > 0))
        print(f"Population density layer stored: "
              f"{populated:,} / {self.nx * self.ny:,} columns populated  "
              f"(max {self.density_map.max():.0f} ppl/km^2)")

    def add_no_fly_zones(self, zones_gdf, min_alt_m=0, max_alt_m=500):
        """Mark CASA-restricted airspace in the grid."""
        print(f"Adding {len(zones_gdf)} no-fly zones...")
        for _, zone in zones_gdf.iterrows():
            minx, miny, maxx, maxy = zone.geometry.bounds
            x0, y0, _ = self.latlon_to_grid(miny, minx, 0)
            x1, y1, _ = self.latlon_to_grid(maxy, maxx, 0)
            z0 = int(min_alt_m / self.voxel_size_m)
            z1 = int(max_alt_m / self.voxel_size_m)
            for x in range(x0, min(x1 + 1, self.nx)):
                for y in range(y0, min(y1 + 1, self.ny)):
                    for z in range(z0, min(z1, self.nz)):
                        self.grid[x, y, z]     = 1
                        self.metadata[x, y, z] = 2

    def add_bankstown_nfz(self):
        """
        Add Bankstown Airport (YBBN) CASA restricted airspace as a no-fly zone.

        Bankstown Airport CTR: Class D controlled airspace, SFC to 1500 ft AGL
        (~457 m). The horizontal boundary is approximately a 5 NM (9.26 km)
        radius circle centred on the aerodrome reference point.

        ARP coordinates: -33.9244 deg S, 150.9883 deg E  (ICAO YSBK)
        Radius used: 9.26 km (5 NM) - conservative; CASA ERSA should be
        consulted for the actual irregular CTR boundary for any real deployment.

        Under CASA CASR Part 101, a remotely piloted aircraft must not operate
        in Class D airspace without ATC clearance.
        """
        arp_lat = -33.9244
        arp_lon = 150.9883
        ctr_radius_m = 2_000.0          # airport perimeter only (~1 NM); medical
                                         # drones obtain ATC clearance, so only the
                                         # immediate runway area needs to be excluded
        ctr_ceiling_m = 457.0           # 1 500 ft AGL ~ 457 m

        min_lon, min_lat, _, _ = self.bounds
        vs = self.voxel_size_m

        # Approximate the circle with a tight grid sweep
        # Degrees of lat/lon per metre
        m_per_lat = self.m_per_deg_lat
        m_per_lon = self.m_per_deg_lon

        radius_lat = ctr_radius_m / m_per_lat
        radius_lon = ctr_radius_m / m_per_lon

        lat0 = arp_lat - radius_lat
        lat1 = arp_lat + radius_lat
        lon0 = arp_lon - radius_lon
        lon1 = arp_lon + radius_lon

        x0, y0, _ = self.latlon_to_grid(lat0, lon0, 0)
        x1, y1, _ = self.latlon_to_grid(lat1, lon1, 0)
        z1 = int(np.ceil(ctr_ceiling_m / vs))

        count = 0
        for xi in range(max(0, x0), min(x1 + 1, self.nx)):
            for yi in range(max(0, y0), min(y1 + 1, self.ny)):
                # Voxel-centre lat/lon
                vlon = min_lon + (xi + 0.5) * vs / m_per_lon
                vlat = min_lat + (yi + 0.5) * vs / m_per_lat
                dx_m = (vlon - arp_lon) * m_per_lon
                dy_m = (vlat - arp_lat) * m_per_lat
                if (dx_m**2 + dy_m**2) <= ctr_radius_m**2:
                    for zi in range(0, min(z1, self.nz)):
                        self.grid[xi, yi, zi]     = 1
                        self.metadata[xi, yi, zi] = 2
                        count += 1

        print(f"Bankstown CTR added: {count:,} voxels marked as no-fly zone "
              f"(radius {ctr_radius_m/1000:.1f} km, ceiling {ctr_ceiling_m:.0f} m)")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath):
        """Pickle the grid to `filepath`."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            'grid':          self.grid,
            'metadata':      self.metadata,
            'bounds':        self.bounds,
            'voxel_size_m':  self.voxel_size_m,
            'max_height_m':  self.max_height_m,
            'dimensions':    (self.nx, self.ny, self.nz),
            'm_per_deg_lat': self.m_per_deg_lat,
            'm_per_deg_lon': self.m_per_deg_lon,
            'density_map':   self.density_map,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        size_mb = os.path.getsize(filepath) / 1_048_576
        print(f"Saved voxel grid to {filepath}  ({size_mb:.2f} MB)")

    @classmethod
    def load(cls, filepath):
        """Reconstruct a VoxelGrid3D from a pickled file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        obj = cls.__new__(cls)
        obj.grid          = data['grid']
        obj.metadata      = data['metadata']
        obj.bounds        = data['bounds']
        obj.voxel_size_m  = data['voxel_size_m']
        obj.max_height_m  = data['max_height_m']
        obj.nx, obj.ny, obj.nz = data['dimensions']
        obj.m_per_deg_lat = data['m_per_deg_lat']
        obj.m_per_deg_lon = data['m_per_deg_lon']
        # Derived dimensions needed by aorva_env._check_termination()
        obj.width_m  = (obj.bounds[2] - obj.bounds[0]) * obj.m_per_deg_lon
        obj.length_m = (obj.bounds[3] - obj.bounds[1]) * obj.m_per_deg_lat
        # Population density layer - zero-filled if not present in old files
        obj.density_map = data.get(
            'density_map',
            np.zeros((obj.nx, obj.ny), dtype=np.float32)
        )
        return obj


# ======================================================================
# Build script  (python voxel_grid_builder.py)
# ======================================================================
if __name__ == "__main__":
    os.makedirs('data',    exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    BOUNDS = (150.8233, -34.0173, 151.0875, -33.7078)  # min_lon, min_lat, max_lon, max_lat

    grid = VoxelGrid3D(bounds=BOUNDS, voxel_size_m=50, max_height_m=500)

    # Try each format that download_buildings.py may have produced
    for candidate in [
        'data/buildings_westmead_liverpool.gpkg',
        'data/buildings_westmead_liverpool.geojson',
        'data/buildings_westmead_liverpool.pkl',
    ]:
        if os.path.exists(candidate):
            print(f"Loading buildings from {candidate}")
            if candidate.endswith('.pkl'):
                with open(candidate, 'rb') as fh:
                    buildings = pickle.load(fh)
            else:
                buildings = gpd.read_file(candidate)
            break
    else:
        raise FileNotFoundError(
            "No buildings file found in data/. Run download_buildings.py first."
        )

    grid.add_buildings(buildings)

    # Add Bankstown Airport controlled airspace (CASA CASR Part 101 compliance)
    grid.add_bankstown_nfz()

    grid.save('data/voxel_grid_westmead_liverpool.pkl')
    print("Done.")
