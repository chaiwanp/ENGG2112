"""
wind_field_interpolator.py

WindField3D: builds a 3D wind field on top of a VoxelGrid3D.

Two population modes
--------------------
1. Freestream / log-law  (original, single observation)
       wind_field = WindField3D(voxel_grid)
       wind_field.set_freestream(speed_ms, direction_deg)
   or:
       wind_field.set_from_dataframe(df, timestamp_idx=-1)

   Horizontally uniform background wind with power-law vertical
   extrapolation from a 10 m reference height.

2. Spatial IDW  (new, multi-node ERA5 data)
       wind_field = WindField3D(voxel_grid, wind_data_df=df)
       wind_field.interpolate_wind_field()

   Inverse-Distance Weighting over a spatial grid of observations
   (e.g. the 6×7 node grid from OpenMeteoWindDownloader).
   Each voxel gets its own wind vector interpolated from nearby nodes.
   Falls back to freestream if only one node is present.

Coordinate conventions
-----------------------
Wind direction is meteorological: the direction FROM which the wind blows,
measured clockwise from North.
    0   = from North  -> flow southward
    90  = from East   -> flow westward
    180 = from South  -> flow northward
    270 = from West   -> flow eastward

Velocity components:
    u = east-west   (+ = eastward, aligns with +x / +longitude)
    v = north-south (+ = northward, aligns with +y / +latitude)
    w = vertical    (+ = upward)

AirSim NED note: NED wind (N, E, D) = this field's (v, u, -w).
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

try:
    from scipy.interpolate import RegularGridInterpolator
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


class WindField3D:
    """
    3D wind field aligned with a VoxelGrid3D.

    Stores u, v, w numpy arrays of shape (nx, ny, nz).
    Queried by sample() (grid coords) or sample_latlon() (world coords).
    """

    def __init__(self, voxel_grid,
                 reference_height_m: float = 10.0,
                 alpha: float = 0.25,
                 wind_data_df: pd.DataFrame | None = None):
        """
        Parameters
        ----------
        voxel_grid : VoxelGrid3D
        reference_height_m : float
            Height of reference wind observation (BoM standard = 10 m).
        alpha : float
            Power-law exponent for vertical extrapolation.
              ~0.14  open terrain / grassland
              ~0.20  small towns
              ~0.25  suburban (Westmead–Liverpool corridor)
              ~0.33  dense urban
        wind_data_df : DataFrame or None
            If provided, used by interpolate_wind_field() for IDW.
            Must have columns: timestamp, wind_speed_kmh,
            wind_direction_deg, lat, lon.
        """
        self.voxel_grid = voxel_grid
        self.z_ref      = float(reference_height_m)
        self.alpha      = float(alpha)
        # Pre-convert timestamps once so interpolate_wind_field() doesn't
        # call df.copy() + pd.to_datetime() on every episode reset.
        if wind_data_df is not None and 'timestamp' in wind_data_df.columns:
            wind_data_df = wind_data_df.copy()
            wind_data_df['timestamp'] = pd.to_datetime(wind_data_df['timestamp'])
        self.wind_data_df = wind_data_df

        shape = (voxel_grid.nx, voxel_grid.ny, voxel_grid.nz)
        self.u_field = np.zeros(shape, dtype=np.float32)
        self.v_field = np.zeros(shape, dtype=np.float32)
        self.w_field = np.zeros(shape, dtype=np.float32)

        # Voxel-centre altitudes for power-law scaling
        self._z_centres = (
            (np.arange(voxel_grid.nz) + 0.5) * voxel_grid.voxel_size_m
        )

        # Lazy scipy interpolators (only used for continuous sampling via sample_latlon)
        self._interp_u = None
        self._interp_v = None
        self._interp_w = None

        # Cache of pre-computed 2D IDW wind maps keyed by timestamp string.
        # Avoids rerunning IDW when the same timestamp is drawn again at reset.
        self._2d_cache: dict = {}

    # ------------------------------------------------------------------
    # Mode 1: freestream / log-law
    # ------------------------------------------------------------------
    def set_freestream(self, wind_speed_ms: float,
                       wind_direction_deg: float) -> None:
        """
        Populate u/v/w with a horizontally-uniform, vertically-scaled
        freestream wind using a power law.

        Parameters
        ----------
        wind_speed_ms : float  speed at reference height (m/s)
        wind_direction_deg : float  meteorological direction (deg)
        """
        theta = np.deg2rad(wind_direction_deg)
        u_ref = -wind_speed_ms * np.sin(theta)
        v_ref = -wind_speed_ms * np.cos(theta)

        z_eff = np.maximum(self._z_centres, 1.0)
        scale = (z_eff / self.z_ref) ** self.alpha   # shape (nz,)

        self.u_field[:] = (u_ref * scale)[np.newaxis, np.newaxis, :]
        self.v_field[:] = (v_ref * scale)[np.newaxis, np.newaxis, :]
        self.w_field[:] = 0.0
        self._invalidate_interpolators()

    def set_from_dataframe(self, wind_df: pd.DataFrame,
                           timestamp_idx: int = -1,
                           speed_col: str = "wind_speed_kmh",
                           direction_col: str = "wind_direction_deg") -> None:
        """
        Pull a single observation from a DataFrame and call set_freestream.
        Speed assumed in km/h (converted to m/s).
        """
        row = wind_df.iloc[timestamp_idx]
        self.set_freestream(
            wind_speed_ms=float(row[speed_col]) / 3.6,
            wind_direction_deg=float(row[direction_col]),
        )

    # ------------------------------------------------------------------
    # Mode 2: spatial IDW interpolation
    # ------------------------------------------------------------------
    def interpolate_wind_field(self, timestamp: pd.Timestamp | None = None,
                                idw_power: float = 2.0) -> None:
        """
        Build the 3D wind field via Inverse Distance Weighting over all
        observation nodes in self.wind_data_df.

        For each horizontal grid position, IDW blends the u/v observations
        from every node weighted by 1/distance^idw_power. The vertical
        dimension then applies the same power-law scaling as set_freestream.

        Falls back to set_from_dataframe (uniform freestream) if only one
        node is present or if wind_data_df is not set.

        Parameters
        ----------
        timestamp : pd.Timestamp or None
            Use observations from this timestamp. None = latest timestamp.
        idw_power : float
            Inverse-distance weighting exponent (default 2).
        """
        df = self.wind_data_df
        if df is None or df.empty:
            print("WindField3D: no wind_data_df set; using zero wind field.")
            return

        # -- Select snapshot (timestamps already converted in __init__) --
        if "timestamp" in df.columns:
            ts = timestamp or df["timestamp"].max()
            snapshot = df[df["timestamp"] == ts]
            if snapshot.empty:
                snapshot = df[df["timestamp"] == df["timestamp"].max()]
        else:
            snapshot = df

        # Deduplicate to one row per node
        snapshot = snapshot.drop_duplicates(subset=["lat", "lon"])

        if len(snapshot) == 0:
            print("WindField3D: no observations in snapshot; using zero field.")
            return

        if len(snapshot) == 1:
            # Only one node -> uniform freestream
            self.set_from_dataframe(snapshot)
            return

        ts_key = str(ts)
        if ts_key in self._2d_cache:
            u_2d, v_2d = self._2d_cache[ts_key]
        else:
            vg = self.voxel_grid
            min_lon, min_lat, _, _ = vg.bounds

            node_gx = ((snapshot["lon"].values - min_lon)
                       * vg.m_per_deg_lon / vg.voxel_size_m)
            node_gy = ((snapshot["lat"].values - min_lat)
                       * vg.m_per_deg_lat / vg.voxel_size_m)

            dirs_rad  = np.deg2rad(snapshot["wind_direction_deg"].values)
            speeds_ms = snapshot["wind_speed_kmh"].values / 3.6
            u_nodes   = -speeds_ms * np.sin(dirs_rad)
            v_nodes   = -speeds_ms * np.cos(dirs_rad)

            xs = np.arange(vg.nx, dtype=np.float32)
            ys = np.arange(vg.ny, dtype=np.float32)
            gx_grid, gy_grid = np.meshgrid(xs, ys, indexing="ij")  # (nx, ny)

            # Fully vectorised IDW — no Python loop over nodes.
            # Shapes: node arrays (n,1,1) broadcast against grid (1,nx,ny).
            dx   = gx_grid[np.newaxis] - node_gx[:, np.newaxis, np.newaxis]  # (n,nx,ny)
            dy   = gy_grid[np.newaxis] - node_gy[:, np.newaxis, np.newaxis]
            dist = np.maximum(np.sqrt(dx**2 + dy**2), 0.5)
            w    = 1.0 / (dist ** idw_power)                                  # (n,nx,ny)
            w_sum = w.sum(axis=0) + 1e-12
            u_2d = (w * u_nodes[:, np.newaxis, np.newaxis]).sum(axis=0) / w_sum
            v_2d = (w * v_nodes[:, np.newaxis, np.newaxis]).sum(axis=0) / w_sum
            u_2d = u_2d.astype(np.float32)
            v_2d = v_2d.astype(np.float32)

            self._2d_cache[ts_key] = (u_2d, v_2d)

        # Apply power-law vertical scaling and broadcast to 3D
        z_eff = np.maximum(self._z_centres, 1.0)
        scale = (z_eff / self.z_ref) ** self.alpha  # (nz,)

        self.u_field = (u_2d[:, :, np.newaxis] * scale[np.newaxis, np.newaxis, :]
                        ).astype(np.float32)
        self.v_field = (v_2d[:, :, np.newaxis] * scale[np.newaxis, np.newaxis, :]
                        ).astype(np.float32)
        self.w_field = np.zeros_like(self.u_field)
        self._invalidate_interpolators()

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def _build_interpolators(self) -> None:
        if not _HAS_SCIPY:
            raise ImportError(
                "scipy is required for continuous sampling. "
                "Install with: pip install scipy"
            )
        xs = np.arange(self.voxel_grid.nx)
        ys = np.arange(self.voxel_grid.ny)
        zs = np.arange(self.voxel_grid.nz)
        self._interp_u = RegularGridInterpolator(
            (xs, ys, zs), self.u_field, bounds_error=False, fill_value=0.0
        )
        self._interp_v = RegularGridInterpolator(
            (xs, ys, zs), self.v_field, bounds_error=False, fill_value=0.0
        )
        self._interp_w = RegularGridInterpolator(
            (xs, ys, zs), self.w_field, bounds_error=False, fill_value=0.0
        )

    def _invalidate_interpolators(self) -> None:
        self._interp_u = None
        self._interp_v = None
        self._interp_w = None

    def sample(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        """
        Sample at a continuous grid position. Returns (u, v, w) in m/s.
        Out-of-bounds queries return (0, 0, 0).
        """
        if self._interp_u is None:
            self._build_interpolators()
        pt = np.array([[x, y, z]])
        return (float(self._interp_u(pt)),
                float(self._interp_v(pt)),
                float(self._interp_w(pt)))

    def sample_latlon(self, lat: float, lon: float,
                      alt_m: float) -> tuple[float, float, float]:
        """Sample at a real-world (lat, lon, alt_m) position."""
        min_lon, min_lat, _, _ = self.voxel_grid.bounds
        vs = self.voxel_grid.voxel_size_m
        x = (lon - min_lon) * self.voxel_grid.m_per_deg_lon / vs
        y = (lat - min_lat) * self.voxel_grid.m_per_deg_lat / vs
        z = alt_m / vs
        return self.sample(x, y, z)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, filepath: str) -> None:
        data = {
            "u_field": self.u_field,
            "v_field": self.v_field,
            "w_field": self.w_field,
            "z_ref":   self.z_ref,
            "alpha":   self.alpha,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved wind field -> {filepath}")

    def load_components(self, filepath: str) -> None:
        """Load u/v/w arrays into this instance. Voxel grid must match."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        expected = (self.voxel_grid.nx,
                    self.voxel_grid.ny,
                    self.voxel_grid.nz)
        if data["u_field"].shape != expected:
            raise ValueError(
                f"Saved shape {data['u_field'].shape} != grid shape {expected}"
            )
        self.u_field = data["u_field"]
        self.v_field = data["v_field"]
        self.w_field = data["w_field"]
        self.z_ref   = data["z_ref"]
        self.alpha   = data["alpha"]
        self._invalidate_interpolators()
        print(f"Loaded wind field <- {filepath}")
