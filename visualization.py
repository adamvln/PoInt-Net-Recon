import numpy as np
import open3d as o3d

def load_point_cloud(filename):
    """
    Load point cloud data from a numpy file.

    Parameters:
    filename (str): Path to the .npy file.

    Returns:
    np.ndarray: Loaded point cloud data.
    """
    point_cloud_data = np.load(filename)
    print(f"Point cloud shape: {point_cloud_data.shape}")
    print(f"First point: {point_cloud_data[0, :]}")
    return point_cloud_data

def create_open3d_point_cloud(point_cloud_data):
    """
    Convert numpy point cloud data to Open3D point cloud.

    Parameters:
    point_cloud_data (np.ndarray): Point cloud data.

    Returns:
    o3d.geometry.PointCloud: Open3D point cloud object.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_data[:, :3])

    # Check if point cloud contains color (RGB) information
    if point_cloud_data.shape[1] >= 6:
        colors = point_cloud_data[:, 3:6]  # Assuming RGB are in columns 4 to 6
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def visualize_point_cloud(pcd):
    """
    Visualize an Open3D point cloud.

    Parameters:
    pcd (o3d.geometry.PointCloud): The Open3D point cloud to visualize.

    Returns:
    None
    """
    o3d.visualization.draw_geometries([pcd])

def main():
    filename = 'dataset/IID_others_/albedo_estimate/final_2452_9708_alb.npy'
    point_cloud_data = load_point_cloud(filename)
    pcd = create_open3d_point_cloud(point_cloud_data)
    visualize_point_cloud(pcd)

if __name__ == "__main__":
    main()
