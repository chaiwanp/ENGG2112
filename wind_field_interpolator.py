"""
wind_field_interpolator.py

WindField3D: 3-D wind field aligned with a VoxelGrid3D.

This is the canonical wind-field module - it merges the simple power-law
implementation (original wind_field_interpolator.py) with the improved
log-law + scipy implementation (3D_wind_field_interpolator.py).

Population methods
------------------
  set_freestream(wind_speed_ms, wind_direction_deg)
      Horizontally-uniform wind scaled by a power-law vertical profile.

  set_from_dataframe(wind_df, timestamp_idx=-1, ...)
      Pull one row from a wind DataFrame and call set_freestream.

  set_from_spatial_observations(observations)
      IDW-interpolate from a list of {lat, lon, speed_ms, direction_deg}
      nodes, then apply the power-law vertically at each grid column.
      This gives a fully spatially-varying 3-D field - different wind at
      each (x, y) column.  Like FEA nodes, but for wind.

  interpolate_wind_field(timestamp=None)
      Auto-detecting wrapper that calls the single-point or spatial path
      depending on whether wind_data has lat/lon columns.

Sampling methods
----------------
  get_wind_at_position(ix, iy, iz)
      Fast integer-index lookup - used by the RL environment every step.

  get_wind_magnitude(ix, iy, iz)
      Scalar wind speed at an integer grid position.

  sample(x, y, z)
      Continuous trilinear interpolation at fractional grid coords (scipy).

  sample_latlon(lat, lon, alt_m)
      Continuous interpolation at real-world coordinates (scipy).

Coordinate conventions
----------------------
  u = east-west   (+ = eastward,  aligns with +x / +longitude)
  v = north-south (+ = northward, aligns with +y / +latitude)
  w = vertical    (+ = upward)

Wind direction is meteorological: the compass bearing FROM which the wind
blows (0 = from North -> flow southward, 90 = from East -> flow westward).

AirSim note: AirSim uses NED. Convert at the integration boundary:
    NED (N, E, D)  <-->  this field (v, u, -w).
"""

import os
import numpy as np
import pickle

try:
    from scipy.interpolate import RegularGridInterpolator
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


class WindField3D:
    """
    3-D wind field on top of a VoxelGrid3D.

    Stores u/v/w arrays of shape (nx, ny, nz).  Populate via
    set_freestream() or set_from_dataframe(), then query via
    get_wind_at_position() (integer, fast) or sample() / sample_latlon()
    (continuous, requires scipy).
    """

    def __init__(self, voxel_grid, wind_data_df=None,
                 reference_height_m=10.0, alpha=0.25):
        """
        Parameters
        ----------
        voxel_grid : VoxelGrid3D
        wind_data_df : pd.DataFrame or None
            Historical observations with at minimum columns
            ['timestamp', 'wind_speed_kmh', 'wind_direction_deg'].
            Only required if you call interpolate_wind_field().
        reference_height_m : float
            BoM standard anemometer height (default 10 m).
        alpha : float
            Power-law exponent for vertical wind profile.
            0.14 = open terrain, 0.25 = suburban (default), 0.33 = dense urban.
        """
        self.voxel_grid = voxel_grid
        self.wind_data  = wind_data_df
        self.z_ref      = float(reference_height_m)
        self.alpha      = float(alpha)

        shape = (voxel_grid.nx, voxel_grid.ny, voxel_grid.nz)
        self.u_field = np.zeros(shape, dtype=np.float32)
        self.v_field = np.zeros(shape, dtype=np.float32)
        self.w_field = np.zeros(shape, dtype=np.float32)

        # Voxel-centre altitudes (m) - used for power-law vertical scaling
        self._z_centres = (
            (np.arange(voxel_grid.nz) + 0.5) * voxel_grid.voxel_size_m
        )

        # Lazy scipy interpolators - built on first sample() call, invalidated
        # whenever the field is repopulated.
        self._interp_u = self._interp_v = self._interp_w = None

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def convert_wind_to_components(speed_kmh, direction_deg):
        """
        Convert a BoM observation to (u, v) at reference height.

        Parameters
        ----------
        speed_kmh : float  - wind speed in km/h
        direction_deg : float  - meteorological direction (degrees from North)

        Returns
        -------
        (u_ms, v_ms) : tuple of floats, m/s
        """
        speed_ms = speed_kmh / 3.6
        theta    = np.radians(direction_deg)
        return float(-speed_ms * np.sin(theta)), float(-speed_ms * np.cos(theta))

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    def set_freestream(self, wind_speed_ms, wind_direction_deg):
        """
        Fill u/v/w with a horizontally-uniform power-law wind profile.

        Parameters
        ----------
        wind_speed_ms : float  - speed at reference height (m/s)
        wind_direction_deg : float  - meteorological direction ( deg from North)
        """
        theta = np.deg2rad(wind_direction_deg)
        u_ref = -wind_speed_ms * np.sin(theta)
        v_ref = -wind_speed_ms * np.cos(theta)

        # Power-law altitude scaling; clamp to >=1 m to avoid log(0)
        z_eff = np.maximum(self._z_centres, 1.0)
        scale = (z_eff / self.z_ref) ** self.alpha    # shape (nz,)

        self.u_field[:] = (u_ref * scale)[np.newaxis, np.newaxis, :]
        self.v_field[:] = (v_ref * scale)[np.newaxis, np.newaxis, :]
        self.w_field[:] = 0.0
        self._invalidate_interpolators()

    def set_from_dataframe(self, wind_df, timestamp_idx=-1,
                           speed_col='wind_speed_kmh',
                           direction_col='wind_direction_deg'):
        """
        Pull one observation from a DataFrame and call set_freestream.

        Parameters
        ----------
        wind_df : pd.DataFrame
        timestamp_idx : int  - row position (default -1 = latest row)
        speed_col : str  - column name for speed in km/h
        direction_col : str  - column name for meteorological direction
        """
        row = wind_df.iloc[timestamp_idx]
        self.set_freestream(
            wind_speed_ms=float(row[speed_col]) / 3.6,
            wind_direction_deg=float(row[direction_col]),
        )

    def set_from_spatial_observations(self, observations, idw_power=2):
        """
        Populate the 3-D wind field from spatially-distributed observations.

        Method
        ------
        Inverse Distance Weighting (IDW) in the horizontal plane:
            u_ref(x, y) = Sigma w_i * u_i  /  Sigma w_i
            w_i = 1 / d(x,y, node_i)^idw_power

        The power-law vertical profile is then applied independently at each
        grid column:
            u(x, y, z) = u_ref(x, y) * (z / z_ref)^alpha

        The whole operation is fully vectorized - no Python loops over the
        (nx, ny) grid.

        Parameters
        ----------
        observations : list of dicts, each with keys:
            lat           float - observation latitude
            lon           float - observation longitude
            speed_ms      float - wind speed at reference height (m/s)
            direction_deg float - meteorological direction ( deg from N)
        idw_power : int or float
            IDW exponent.  2 (inverse-square) is standard.

        Notes
        -----
        Memory: O(nx * ny * n_obs) floats.  For a 489x690 grid and 12
        observation nodes this is ~16 MB - well within typical limits.
        """
        if not observations:
            raise ValueError("observations list is empty")

        n_obs    = len(observations)
        obs_lats = np.array([o['lat']                               for o in observations])
        obs_lons = np.array([o['lon']                               for o in observations])
        # Convert (speed, met-direction) -> (u, v) components
        obs_u    = np.array([-o['speed_ms'] * np.sin(np.deg2rad(o['direction_deg']))
                              for o in observations])
        obs_v    = np.array([-o['speed_ms'] * np.cos(np.deg2rad(o['direction_deg']))
                              for o in observations])

        # -- Grid column centres: (nx, ny) ------------------------------
        min_lon, min_lat, _, _ = self.voxel_grid.bounds
        vs  = self.voxel_grid.voxel_size_m
        IX, IY = np.meshgrid(
            np.arange(self.voxel_grid.nx),
            np.arange(self.voxel_grid.ny),
            indexing='ij',
        )  # (nx, ny)
        grid_lons = min_lon + (IX + 0.5) * vs / self.voxel_grid.m_per_deg_lon
        grid_lats = min_lat + (IY + 0.5) * vs / self.voxel_grid.m_per_deg_lat

        # -- IDW weights: (nx, ny, n_obs) -------------------------------
        # Distances in degrees (valid proxy for ~25 km box; no projection needed)
        dlat = grid_lats[:, :, np.newaxis] - obs_lats[np.newaxis, np.newaxis, :]
        dlon = grid_lons[:, :, np.newaxis] - obs_lons[np.newaxis, np.newaxis, :]
        d2   = dlat**2 + dlon**2  # (nx, ny, n_obs)

        # Add epsilon so a column sitting exactly on a node still works
        weights = 1.0 / (d2 + 1e-14) ** (idw_power / 2.0)
        weights /= weights.sum(axis=2, keepdims=True)   # normalise -> sum = 1

        # -- Interpolated reference-height wind: (nx, ny) ---------------
        u_ref = (weights * obs_u[np.newaxis, np.newaxis, :]).sum(axis=2)
        v_ref = (weights * obs_v[np.newaxis, np.newaxis, :]).sum(axis=2)

        # -- Power-law vertical scaling: (nz,) --------------------------
        scale = (np.maximum(self._z_centres, 1.0) / self.z_ref) ** self.alpha

        # -- Fill 3-D arrays: u[ix, iy, iz] = u_ref[ix, iy] x scale[iz] -
        self.u_field[:] = u_ref[:, :, np.newaxis] * scale[np.newaxis, np.newaxis, :]
        self.v_field[:] = v_ref[:, :, np.newaxis] * scale[np.newaxis, np.newaxis, :]
        self.w_field[:] = 0.0
        self._invalidate_interpolators()

        spd_mean = float(np.sqrt(u_ref**2 + v_ref**2).mean())
        print(f"Spatial wind field set: {n_obs} nodes, "
              f"mean ref-height speed = {spd_mean:.2f} m/s")

    def _interpolate_spatial(self, timestamp=None):
        """
        Populate the field from self.wind_data when it contains lat/lon columns
        (i.e. a multi-node spatial dataset from download_spatial_grid()).

        Selects all rows matching the nearest timestamp and calls
        set_from_spatial_observations().
        """
        times = self.wind_data['timestamp'].drop_duplicates().sort_values()
        if timestamp is None:
            t_sel = times.iloc[-1]
        else:
            idx   = (times - timestamp).abs().idxmin()
            t_sel = times.loc[idx]

        rows = self.wind_data[self.wind_data['timestamp'] == t_sel]
        observations = [
            {
                'lat':           float(r['lat']),
                'lon':           float(r['lon']),
                'speed_ms':      float(r['wind_speed_kmh']) / 3.6,
                'direction_deg': float(r['wind_direction_deg']),
            }
            for _, r in rows.iterrows()
        ]
        self.set_from_spatial_observations(observations)

    def interpolate_wind_field(self, timestamp=None):
        """
        Populate the field from self.wind_data.

        Auto-detects the data format:
          - If wind_data has 'lat' and 'lon' columns -> spatially-varying IDW
            interpolation via set_from_spatial_observations().
          - Otherwise -> single-point uniform field via set_freestream().

        Parameters
        ----------
        timestamp : pd.Timestamp or None
            Target time.  None uses the latest available observation.

        Requires wind_data_df to have been passed to the constructor.
        """
        if self.wind_data is None:
            raise RuntimeError(
                "wind_data is not set. Pass wind_data_df to the constructor, "
                "or call set_from_dataframe() / set_freestream() directly."
            )

        is_spatial = ('lat' in self.wind_data.columns and
                      'lon' in self.wind_data.columns)

        if is_spatial:
            self._interpolate_spatial(timestamp)
        else:
            # Single-point path (original behaviour)
            if timestamp is None:
                pos = -1
                row = self.wind_data.iloc[-1]
            else:
                label = (self.wind_data['timestamp'] - timestamp).abs().idxmin()
                pos   = self.wind_data.index.get_loc(label)
                row   = self.wind_data.iloc[pos]

            self.set_from_dataframe(self.wind_data, timestamp_idx=pos)
            spd  = float(row.get('wind_speed_kmh',     0))
            dirn = float(row.get('wind_direction_deg', 0))
            print(f"Wind field set: {spd:.1f} km/h from {dirn:.0f} deg")

    # ------------------------------------------------------------------
    # Sampling - integer (fast, no scipy required)
    # ------------------------------------------------------------------

    def get_wind_at_position(self, x, y, z):
        """
        Wind at an integer grid index, clamped to valid range.

        Returns
        -------
        (u, v, w) : tuple of floats, m/s
        """
        x = int(np.clip(x, 0, self.voxel_grid.nx - 1))
        y = int(np.clip(y, 0, self.voxel_grid.ny - 1))
        z = int(np.clip(z, 0, self.voxel_grid.nz - 1))
        return (float(self.u_field[x, y, z]),
                float(self.v_field[x, y, z]),
                float(self.w_field[x, y, z]))

    def get_wind_magnitude(self, x, y, z):
        """Scalar wind speed at an integer grid position (m/s)."""
        u, v, w = self.get_wind_at_position(x, y, z)
        return float(np.sqrt(u**2 + v**2 + w**2))

    # ------------------------------------------------------------------
    # Sampling - continuous trilinear (requires scipy)
    # ------------------------------------------------------------------

    def _build_interpolators(self):
        if not _HAS_SCIPY:
            raise ImportError(
                "scipy is required for continuous wind sampling. "
                "Install with: pip install scipy"
            )
        xs = np.arange(self.voxel_grid.nx)
        ys = np.arange(self.voxel_grid.ny)
        zs = np.arange(self.voxel_grid.nz)
        kw = dict(bounds_error=False, fill_value=0.0)
        self._interp_u = RegularGridInterpolator((xs, ys, zs), self.u_field, **kw)
        self._interp_v = RegularGridInterpolator((xs, ys, zs), self.v_field, **kw)
        self._interp_w = RegularGridInterpolator((xs, ys, zs), self.w_field, **kw)

    def _invalidate_interpolators(self):
        self._interp_u = self._interp_v = self._interp_w = None

    def sample(self, x, y, z):
        """
        Continuous trilinear interpolation at fractional grid coordinates.

        Out-of-bounds positions return (0, 0, 0).

        Returns (u, v, w) in m/s.
        """
        if self._interp_u is None:
            self._build_interpolators()
        pt = np.array([[x, y, z]])
        return (float(self._interp_u(pt)),
                float(self._interp_v(pt)),
                float(self._interp_w(pt)))

    def sample_latlon(self, lat, lon, alt_m):
        """
        Continuous interpolation at real-world (lat, lon, alt_m).

        Useful for the AirSim integration layer after converting drone pose.
        """
        min_lon, min_lat, _, _ = self.voxel_grid.bounds
        vs = self.voxel_grid.voxel_size_m
        x  = (lon - min_lon) * self.voxel_grid.m_per_deg_lon / vs
        y  = (lat - min_lat) * self.voxel_grid.m_per_deg_lat / vs
        z  = alt_m / vs
        return self.sample(x, y, z)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath):
        """Pickle u/v/w arrays and parameters to `filepath`."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            'u_field': self.u_field,
            'v_field': self.v_field,
            'w_field': self.w_field,
            'z_ref':   self.z_ref,
            'alpha':   self.alpha,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved wind field to {filepath}")

    def load_components(self, filepath):
        """
        Load pre-computed u/v/w arrays into this instance.

        The voxel grid dimensions must match those used when saving.
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        expected = (self.voxel_grid.nx, self.voxel_grid.ny, self.voxel_grid.nz)
        if data['u_field'].shape != expected:
            raise ValueError(
                f"Saved wind field shape {data['u_field'].shape} does not match "
                f"voxel grid shape {expected}"
            )
        self.u_field = data['u_field']
        self.v_field = data['v_field']
        self.w_field = data['w_field']
        self.z_ref   = data['z_ref']
        self.alpha   = data['alpha']
        self._invalidate_interpolators()
        print(f"Loaded wind field from {filepath}")


# ======================================================================
# Verification demo  (python wind_field_interpolator.py)
# ======================================================================
if __name__ == "__main__":
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from voxel_grid_builder import VoxelGrid3D

    os.makedirs('data',    exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    voxel_grid = VoxelGrid3D.load('data/voxel_grid_westmead_liverpool.pkl')

    # Prefer real spatial wind data; fall back to single-point real data
    wind_source = None
    for csv_path in ['data/wind_spatial_real.csv', 'data/wind_historical_real.csv']:
        if os.path.exists(csv_path):
            wind_source = csv_path
            break

    if wind_source is None:
        print("No wind CSV found. Run  python scripts/00_download_real_wind.py  first.")
        raise SystemExit(1)

    print(f"Loading wind data from {wind_source}")
    wind_df = pd.read_csv(wind_source)
    wind_df['timestamp'] = pd.to_datetime(wind_df['timestamp'])

    wf = WindField3D(voxel_grid, wind_data_df=wind_df)
    wf.interpolate_wind_field()
    wf.save('data/wind_field_current.pkl')

    # --- Power-law altitude verification ---
    print("\n=== Power-law altitude verification (alpha={:.2f}) ===".format(wf.alpha))
    print(f"  {'Alt (m)':>7}  {'Max speed (m/s)':>15}  {'Scale factor':>12}")
    for alt_m in [10, 50, 100, 150, 200, 300, 400]:
        iz    = min(int(alt_m / voxel_grid.voxel_size_m), voxel_grid.nz - 1)
        speed = float(np.sqrt(wf.u_field[:, :, iz]**2 +
                               wf.v_field[:, :, iz]**2).max())
        ratio = (max(alt_m, 1.0) / wf.z_ref) ** wf.alpha
        print(f"  {alt_m:7d}  {speed:15.3f}  {ratio:12.4f}")

    u, v, w = wf.sample_latlon(-33.8078, 150.9875, 100.0)
    print(f"\nAt Westmead 100 m: u={u:.2f}  v={v:.2f}  w={w:.2f}  "
          f"speed={np.sqrt(u**2 + v**2):.2f} m/s")

    # --- Improved quiver plots at three altitudes ---
    for alt_m in [50, 100, 200]:
        iz  = min(int(alt_m / voxel_grid.voxel_size_m), voxel_grid.nz - 1)

        U_full = wf.u_field[:, :, iz]
        V_full = wf.v_field[:, :, iz]
        speed_map = np.sqrt(U_full**2 + V_full**2)

        skip = max(1, voxel_grid.nx // 30)
        X, Y = np.meshgrid(
            np.arange(0, voxel_grid.nx, skip),
            np.arange(0, voxel_grid.ny, skip),
        )
        U = U_full[::skip, ::skip].T
        V = V_full[::skip, ::skip].T

        fig = plt.figure(figsize=(14, 10))
        gs  = gridspec.GridSpec(1, 2, width_ratios=[3, 1], figure=fig)
        ax_map  = fig.add_subplot(gs[0])
        ax_prof = fig.add_subplot(gs[1])

        # Speed heatmap
        im = ax_map.imshow(speed_map.T, origin='lower', cmap='Blues', alpha=0.7,
                           extent=[0, voxel_grid.nx, 0, voxel_grid.ny],
                           aspect='auto')
        plt.colorbar(im, ax=ax_map, label='Wind speed (m/s)',
                     fraction=0.03, pad=0.02)

        # Wind arrows
        Q = ax_map.quiver(X, Y, U, V,
                          np.sqrt(U**2 + V**2), cmap='coolwarm',
                          pivot='mid', width=0.003, headwidth=4, scale=None)

        # Hospitals
        wx, wy, _ = voxel_grid.latlon_to_grid(-33.8078, 150.9875, 0)
        lx, ly, _ = voxel_grid.latlon_to_grid(-33.9173, 150.9233, 0)
        ax_map.plot(wx, wy, 'b*', markersize=20, markeredgecolor='white',
                    markeredgewidth=1.5, label='Westmead', zorder=9)
        ax_map.plot(lx, ly, 'g*', markersize=20, markeredgecolor='white',
                    markeredgewidth=1.5, label='Liverpool', zorder=9)
        ax_map.plot([wx, lx], [wy, ly], 'k--', linewidth=1.2,
                    alpha=0.5, label='Corridor', zorder=5)

        ax_map.set_xlabel('Grid X  (west -> east)', fontsize=9)
        ax_map.set_ylabel('Grid Y  (south -> north)', fontsize=9)
        ax_map.set_title(
            f'Wind field at {alt_m} m altitude\n'
            f'mean={speed_map.mean():.2f} m/s  max={speed_map.max():.2f} m/s  '
            f'alpha={wf.alpha}',
            fontsize=10,
        )
        ax_map.legend(loc='upper right', fontsize=9, framealpha=0.9)
        ax_map.grid(True, alpha=0.15)

        # Right panel: altitude wind-speed profile
        alts_m = np.linspace(10, voxel_grid.max_height_m, 200)
        scale  = (np.maximum(alts_m, 1.0) / wf.z_ref) ** wf.alpha
        ref_spd = float(speed_map[voxel_grid.nx // 2, voxel_grid.ny // 2])
        ax_prof.plot(ref_spd * scale, alts_m, '-', color='steelblue', linewidth=2)
        ax_prof.axhline(alt_m, color='red', linestyle='--', linewidth=1.2,
                        label=f'Current slice: {alt_m} m')
        ax_prof.set_xlabel('Wind speed (m/s)', fontsize=9)
        ax_prof.set_ylabel('Altitude (m)', fontsize=9)
        ax_prof.set_title(f'Power-law profile\n(alpha={wf.alpha})', fontsize=9)
        ax_prof.legend(fontsize=8)
        ax_prof.grid(True, alpha=0.3)

        plt.tight_layout()
        out = f'outputs/wind_field_{alt_m}m.png'
        plt.savefig(out, dpi=250, bbox_inches='tight')
        plt.close()
        print(f"Saved {out}")

    print("\nDone.")
