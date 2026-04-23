"""
wind_field_interpolator.py

WindField3D: builds a 3D wind field on top of a VoxelGrid3D by log-law
extrapolation from a single reference-height observation (e.g. BoM at 10 m).

This is the freestream / background wind only. Urban modification
(canyon channelling, building wakes, corner acceleration) is layered on
top in a separate pass.

Coordinate conventions
----------------------
Wind direction is meteorological: the direction FROM which the wind blows,
measured clockwise from North, in degrees.
    0   = wind from North  -> flow southward
    90  = wind from East   -> flow westward
    180 = wind from South  -> flow northward
    270 = wind from West   -> flow eastward

Velocity components:
    u = east-west   (+ = eastward, aligns with +x / +longitude)
    v = north-south (+ = northward, aligns with +y / +latitude)
    w = vertical    (+ = upward)

Note for AirSim integration: AirSim uses NED (North-East-Down). A NED wind
vector (N, E, D) corresponds to this field's (v, u, -w). Do the transform
at the AirSim boundary, not in this class.
"""

import numpy as np
import pickle

try:
    from scipy.interpolate import RegularGridInterpolator
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


class WindField3D:
    """
    3D wind field aligned with a VoxelGrid3D.

    Stores three numpy arrays of shape (nx, ny, nz) -- one per velocity
    component. Populated by `set_freestream` or `set_from_dataframe`, then
    queried by `sample` (grid coords) or `sample_latlon` (world coords).
    """

    def __init__(self, voxel_grid, reference_height_m=10.0, alpha=0.25):
        """
        Parameters
        ----------
        voxel_grid : VoxelGrid3D
            The spatial grid to populate. Must have attributes:
            nx, ny, nz, voxel_size_m, bounds, m_per_deg_lat, m_per_deg_lon.
        reference_height_m : float
            Height of the reference wind observation (default 10 m, which
            is the BoM standard anemometer height).
        alpha : float
            Power-law exponent for vertical extrapolation.
                ~0.14  open terrain / grassland
                ~0.20  small towns
                ~0.25  suburban (Westmead-Liverpool corridor)
                ~0.33  dense urban
        """
        self.voxel_grid = voxel_grid
        self.z_ref = float(reference_height_m)
        self.alpha = float(alpha)

        shape = (voxel_grid.nx, voxel_grid.ny, voxel_grid.nz)
        self.u_field = np.zeros(shape, dtype=np.float32)
        self.v_field = np.zeros(shape, dtype=np.float32)
        self.w_field = np.zeros(shape, dtype=np.float32)

        # Voxel-centre altitudes (m) for log-law scaling
        self._z_centres = (np.arange(voxel_grid.nz) + 0.5) * voxel_grid.voxel_size_m

        # Interpolators built lazily on first sample() call
        self._interp_u = None
        self._interp_v = None
        self._interp_w = None

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------
    def set_freestream(self, wind_speed_ms, wind_direction_deg):
        """
        Populate u, v, w arrays with a horizontally-uniform,
        vertically-log-law freestream wind.

        Parameters
        ----------
        wind_speed_ms : float
            Wind speed at reference height, in metres per second.
        wind_direction_deg : float
            Meteorological direction (degrees clockwise from North).
        """
        theta = np.deg2rad(wind_direction_deg)

        # Convert "from" direction to flow vector components at reference height
        u_ref = -wind_speed_ms * np.sin(theta)   # east-west
        v_ref = -wind_speed_ms * np.cos(theta)   # north-south

        # Log-law altitude scaling; clamp z to prevent division by zero
        z_eff = np.maximum(self._z_centres, 1.0)
        scale = (z_eff / self.z_ref) ** self.alpha            # shape (nz,)

        # Broadcast across (nx, ny, nz)
        self.u_field[:] = (u_ref * scale)[np.newaxis, np.newaxis, :]
        self.v_field[:] = (v_ref * scale)[np.newaxis, np.newaxis, :]
        self.w_field[:] = 0.0

        self._invalidate_interpolators()

    def set_from_dataframe(self, wind_df, timestamp_idx=-1,
                           speed_col='wind_speed_kmh',
                           direction_col='wind_direction_deg'):
        """
        Convenience: pull a single observation from a wind DataFrame and
        populate the field.

        Assumes speed is in km/h (converts to m/s) and direction is
        meteorological degrees. Override via speed_col/direction_col if
        your DataFrame uses different names.
        """
        row = wind_df.iloc[timestamp_idx]
        speed_ms = float(row[speed_col]) / 3.6
        direction_deg = float(row[direction_col])
        self.set_freestream(speed_ms, direction_deg)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def _build_interpolators(self):
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

    def _invalidate_interpolators(self):
        self._interp_u = None
        self._interp_v = None
        self._interp_w = None

    def sample(self, x, y, z):
        """
        Sample the wind field at a continuous grid position.

        Parameters
        ----------
        x, y, z : float
            Fractional grid coordinates. Out-of-bounds queries return zero.

        Returns
        -------
        (u, v, w) : tuple of floats, in m/s
        """
        if self._interp_u is None:
            self._build_interpolators()

        pt = np.array([[x, y, z]])
        return (float(self._interp_u(pt)),
                float(self._interp_v(pt)),
                float(self._interp_w(pt)))

    def sample_latlon(self, lat, lon, alt_m):
        """
        Sample at a real-world (lat, lon, alt) position. Use this from the
        AirSim integration layer after converting the drone pose.
        """
        min_lon, min_lat, _, _ = self.voxel_grid.bounds
        vs = self.voxel_grid.voxel_size_m

        x = (lon - min_lon) * self.voxel_grid.m_per_deg_lon / vs
        y = (lat - min_lat) * self.voxel_grid.m_per_deg_lat / vs
        z = alt_m / vs

        return self.sample(x, y, z)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, filepath):
        """Pickle the three component arrays + parameters."""
        data = {
            'u_field': self.u_field,
            'v_field': self.v_field,
            'w_field': self.w_field,
            'z_ref': self.z_ref,
            'alpha': self.alpha,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved wind field to {filepath}")

    def load_components(self, filepath):
        """
        Load previously-saved u/v/w arrays into this instance. The voxel
        grid must match the one used when saving.
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        expected_shape = (self.voxel_grid.nx, self.voxel_grid.ny, self.voxel_grid.nz)
        if data['u_field'].shape != expected_shape:
            raise ValueError(
                f"Saved wind field shape {data['u_field'].shape} does not match "
                f"voxel grid shape {expected_shape}"
            )

        self.u_field = data['u_field']
        self.v_field = data['v_field']
        self.w_field = data['w_field']
        self.z_ref = data['z_ref']
        self.alpha = data['alpha']
        self._invalidate_interpolators()
        print(f"Loaded wind field from {filepath}")


# ======================================================================
# Demo / verification script
# ======================================================================
if __name__ == "__main__":
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    from voxel_grid_builder import VoxelGrid3D   # your existing module

    # --- Load existing artefacts ---
    voxel_grid = VoxelGrid3D.load('data/voxel_grid_westmead_liverpool.pkl')
    wind_df = pd.read_csv('data/wind_historical_synthetic.csv')

    # --- Build the wind field from a single observation ---
    wind_field = WindField3D(voxel_grid, reference_height_m=10.0, alpha=0.25)
    wind_field.set_from_dataframe(wind_df, timestamp_idx=-1)

    # --- Verify log-law scaling ---
    # Max horizontal wind speed at a few altitudes. Ratio should match alpha law.
    print("\n=== Log-law verification ===")
    for alt_m in [10, 50, 100, 200, 400]:
        z_idx = int(alt_m / voxel_grid.voxel_size_m)
        z_idx = min(z_idx, voxel_grid.nz - 1)
        u_slice = wind_field.u_field[:, :, z_idx]
        v_slice = wind_field.v_field[:, :, z_idx]
        speed = np.sqrt(u_slice**2 + v_slice**2).max()
        expected_ratio = (max(alt_m, 1.0) / 10.0) ** 0.25
        print(f"  alt = {alt_m:4d} m  |  max speed = {speed:5.2f} m/s  "
              f"|  ratio vs 10 m = {expected_ratio:.3f}")

    # --- Sample at a point (Westmead Hospital, 100 m alt) ---
    u, v, w = wind_field.sample_latlon(lat=-33.8078, lon=150.9875, alt_m=100)
    speed = np.sqrt(u**2 + v**2)
    print(f"\nSample at Westmead (100 m): u={u:.2f}, v={v:.2f}, w={w:.2f}  "
          f"|speed|={speed:.2f} m/s")

    # --- Persist for the visualiser ---
    os.makedirs('data', exist_ok=True)
    wind_field.save('data/wind_field_current.pkl')

    # --- Visualise at 3 altitudes ---
    os.makedirs('outputs', exist_ok=True)
    skip = 5
    for alt_m in [50, 100, 200]:
        z_idx = min(int(alt_m / voxel_grid.voxel_size_m), voxel_grid.nz - 1)
        u_slice = wind_field.u_field[:, :, z_idx]
        v_slice = wind_field.v_field[:, :, z_idx]

        X, Y = np.meshgrid(np.arange(0, voxel_grid.nx, skip),
                           np.arange(0, voxel_grid.ny, skip))
        U = u_slice[::skip, ::skip].T
        V = v_slice[::skip, ::skip].T
        mag = np.sqrt(U**2 + V**2)

        fig, ax = plt.subplots(figsize=(12, 10))
        Q = ax.quiver(X, Y, U, V, mag, cmap='coolwarm', scale=80)
        ax.set_xlabel('Grid X')
        ax.set_ylabel('Grid Y')
        ax.set_title(f'Freestream wind field at {alt_m} m (log-law, alpha=0.25)')
        plt.colorbar(Q, ax=ax, label='Wind speed (m/s)')
        plt.tight_layout()
        plt.savefig(f'outputs/wind_field_{alt_m}m.png', dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved outputs/wind_field_{alt_m}m.png")

    print("\nDone.")
