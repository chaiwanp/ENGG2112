# visualize_voxel_grid.py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def visualize_voxel_slice(voxel_grid, altitude_m=100):
    """Visualize horizontal slice at specific altitude"""
    
    z_idx = int(altitude_m / voxel_grid.voxel_size_m)
    
    slice_data = voxel_grid.grid[:, :, z_idx]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(slice_data.T, origin='lower', cmap='RdYlGn_r', 
                   extent=[0, voxel_grid.nx, 0, voxel_grid.ny])
    
    # Mark hospitals
    westmead_x, westmead_y, _ = voxel_grid.latlon_to_grid(-33.8078, 150.9875, 0)
    liverpool_x, liverpool_y, _ = voxel_grid.latlon_to_grid(-33.9173, 150.9233, 0)
    
    ax.plot(westmead_x, westmead_y, 'b*', markersize=20, label='Westmead Hospital')
    ax.plot(liverpool_x, liverpool_y, 'r*', markersize=20, label='Liverpool Hospital')
    
    ax.set_xlabel('Grid X')
    ax.set_ylabel('Grid Y')
    ax.set_title(f'Voxel Grid at {altitude_m}m Altitude\n(Red=Occupied, Green=Free)')
    ax.legend()
    
    plt.colorbar(im, ax=ax, label='Occupancy')
    plt.tight_layout()
    plt.savefig(f'outputs/voxel_slice_{altitude_m}m.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_wind_field(wind_field, altitude_m=100):
    """Visualize wind field at specific altitude"""
    
    z_idx = int(altitude_m / wind_field.voxel_grid.voxel_size_m)
    
    u_slice = wind_field.u_field[:, :, z_idx]
    v_slice = wind_field.v_field[:, :, z_idx]
    
    # Downsample for visualization
    skip = 5
    X, Y = np.meshgrid(
        range(0, wind_field.voxel_grid.nx, skip),
        range(0, wind_field.voxel_grid.ny, skip)
    )
    
    U = u_slice[::skip, ::skip].T
    V = v_slice[::skip, ::skip].T
    
    magnitude = np.sqrt(U**2 + V**2)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot wind vectors
    Q = ax.quiver(X, Y, U, V, magnitude, cmap='coolwarm', scale=50)
    
    ax.set_xlabel('Grid X')
    ax.set_ylabel('Grid Y')
    ax.set_title(f'Wind Field at {altitude_m}m Altitude')
    plt.colorbar(Q, ax=ax, label='Wind Speed (m/s)')
    plt.tight_layout()
    plt.savefig(f'outputs/wind_field_{altitude_m}m.png', dpi=300, bbox_inches='tight')
    plt.show()

# Load and visualize
voxel_grid = VoxelGrid3D.load('data/voxel_grid_westmead_liverpool.pkl')
visualize_voxel_slice(voxel_grid, altitude_m=100)
visualize_voxel_slice(voxel_grid, altitude_m=200)

# Visualize wind
with open('data/wind_field_current.pkl', 'rb') as f:
    wind_data = pickle.load(f)

wind_field = WindField3D(voxel_grid, wind_df)
wind_field.u_field = wind_data['u_field']
wind_field.v_field = wind_data['v_field']
wind_field.w_field = wind_data['w_field']

visualize_wind_field(wind_field, altitude_m=100)