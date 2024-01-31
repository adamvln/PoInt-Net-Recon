import numpy as np
import open3d as o3d
import laspy
import os
from scipy.spatial import cKDTree

def voxel_downsample_point_cloud_only_rgb(np_file_path, voxel_size):
    """
    Downsamples a point cloud stored in a NumPy file using voxel downsampling.
    This point cloud includes additional attributes such as RGB.

    Parameters:
    - np_file_path: The file path to the NumPy file containing the point cloud data.
    - voxel_size: The size of the voxel to downsample the point cloud.

    Returns:
    - downsampled_point_cloud_np: The downsampled point cloud as a NumPy array.
    """

    # Load the point cloud data from a NumPy file
    point_cloud_np = np.load(np_file_path)

    # Separate XYZ coordinates and additional attributes
    xyz = point_cloud_np[:, :3]
    additional_attributes = point_cloud_np[:, 3:6]

    # Convert to Open3D point cloud using only XYZ
    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(xyz)

    # Perform voxel downsampling
    downsampled_point_cloud_o3d = point_cloud_o3d.voxel_down_sample(voxel_size)

    # Convert back to NumPy array (XYZ only)
    downsampled_xyz_np = np.asarray(downsampled_point_cloud_o3d.points)

    # Create a KDTree for the original point cloud
    tree = cKDTree(xyz)

    # For each downsampled point, find its nearest neighbor in the original cloud
    _, indices = tree.query(downsampled_xyz_np)

    # Use these indices to gather the corresponding additional attributes
    downsampled_attributes = additional_attributes[indices]

    # Combine the downsampled XYZ with the corresponding attributes
    downsampled_point_cloud_np = np.hstack((downsampled_xyz_np, downsampled_attributes))

    return downsampled_point_cloud_np

def voxel_downsample_point_cloud(np_file_path, voxel_size):
    """
    Downsamples a point cloud stored in a NumPy file using voxel downsampling.
    This point cloud includes additional attributes such as RGB and intensity.

    Parameters:
    - np_file_path: The file path to the NumPy file containing the point cloud data.
    - voxel_size: The size of the voxel to downsample the point cloud.

    Returns:
    - downsampled_point_cloud_np: The downsampled point cloud as a NumPy array.
    """

    # Load the point cloud data from a NumPy file
    point_cloud_np = np.load(np_file_path)

    # Separate XYZ coordinates and additional attributes
    xyz = point_cloud_np[:, :3]
    additional_attributes = point_cloud_np[:, 3:]

    # Convert to Open3D point cloud using only XYZ
    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(xyz)

    # Perform voxel downsampling
    downsampled_point_cloud_o3d = point_cloud_o3d.voxel_down_sample(voxel_size)

    # Convert back to NumPy array (XYZ only)
    downsampled_xyz_np = np.asarray(downsampled_point_cloud_o3d.points)

    # Create a KDTree for the original point cloud
    tree = cKDTree(xyz)

    # For each downsampled point, find its nearest neighbor in the original cloud
    _, indices = tree.query(downsampled_xyz_np)

    # Use these indices to gather the corresponding additional attributes
    downsampled_attributes = additional_attributes[indices]

    # Combine the downsampled XYZ with the corresponding attributes
    downsampled_point_cloud_np = np.hstack((downsampled_xyz_np, downsampled_attributes))

    return downsampled_point_cloud_np

def laz_to_numpy(laz_file_path):
    """
    Convert a LAZ file to a NumPy array.

    Parameters:
    laz_file_path: str
        The file path to the LAZ file.

    Returns:
    np.ndarray
        A NumPy array containing the point cloud data.
    """
    # Read the .laz file
    las = laspy.read(laz_file_path)

    # Extracting point cloud data
    point_cloud_data = np.vstack((las.x, las.y, las.z)).transpose()

    # Check for additional attributes like color and intensity
    if np.all([hasattr(las, attr) for attr in ["red", "green", "blue"]]):
        colors = np.vstack((las.red, las.green, las.blue)).transpose()
        point_cloud_data = np.hstack((point_cloud_data, colors))

    if hasattr(las, 'intensity'):
        intensity = las.intensity[:, np.newaxis]
        point_cloud_data = np.hstack((point_cloud_data, intensity))

    return point_cloud_data

def compute_normals(np_file_path, search_param):
    """
    Computes normals of a point cloud stored in a NumPy file using Open3D.

    Parameters:
    - np_file_path: The file path to the NumPy file containing the point cloud data.
    - search_param: The search parameters for normal estimation (e.g., radius or KNN).

    Returns:
    - normals_np: The normals of the point cloud as a NumPy array.
    """

    # Load the point cloud data from a NumPy file
    point_cloud_np = np.load(np_file_path)

    # Convert to Open3D point cloud
    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud_np[:, :3])

    # Estimate normals
    point_cloud_o3d.estimate_normals(search_param=search_param)

    # Optionally orient the normals (if necessary for your application)
    # point_cloud_o3d.orient_normals_consistent_tangent_plane(k)

    # Convert normals to NumPy array
    normals_np = np.asarray(point_cloud_o3d.normals)

    return normals_np

def normalize_point_cloud(point_cloud):
    """
    Normalize a point cloud so that it fits within the range of 0 to 1.

    Parameters:
    - point_cloud: A NumPy array of shape (N, 3) representing the point cloud.

    Returns:
    - normalized_point_cloud: The normalized point cloud as a NumPy array.
    """

    # Find the minimum and maximum values along each axis
    min_values = np.min(point_cloud, axis=0)
    max_values = np.max(point_cloud, axis=0)

    # Translate the point cloud to align the minimum point with the origin
    translated_point_cloud = point_cloud - min_values

    # Find the maximum extent of the translated point cloud
    max_extent = np.max(translated_point_cloud, axis=0)

    # Scale the point cloud to fit within the range of 0 to 1
    normalized_point_cloud = translated_point_cloud / max_extent

    return normalized_point_cloud

def transform_normals_to_0_255(normals):
    # Normalize from -1 to 1 to 0 to 1
    normalized = (normals + 1) / 2

    # Scale from 0 to 1 to 0 to 255
    transformed = normalized * 255

    return transformed

def main_downsampling():
    voxel_size = 0.1  # Set the voxel size

    # Define the input and output directories
    input_dir = 'Data/npy_pc'
    output_dir = f'Data/npy_down_onlyrgb_{voxel_size}'

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over all .npy files in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.npy'):
            print(f"Processing {file_name}...")
            # Construct full file paths
            input_file_path = os.path.join(input_dir, file_name)
            output_file_path = os.path.join(output_dir, file_name)

            # Perform voxel downsampling
            downsampled_pc = voxel_downsample_point_cloud_only_rgb(input_file_path, voxel_size)

            # Save the downsampled point cloud to the output directory
            np.save(output_file_path, downsampled_pc)

    print(f"All point clouds have been downsampled with voxel size {voxel_size} and saved to {output_dir}.")

def main_laz2npy():
    # Define the input and output directories
    input_dir = 'Data/laz_pc'
    output_dir = 'Data/pcd/pcd_from_laz'

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over all .npy files in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.laz'):
            print(f"Processing {file_name}...")
            # Construct full file paths
            input_file_path = os.path.join(input_dir, file_name)
            
            filename_without_extension = os.path.splitext(file_name)[0]
            output_file_path = os.path.join(output_dir, filename_without_extension)
            # Perform point cloud conversion
            npy_pc = laz_to_numpy(input_file_path)

            # Save the downsampled point cloud to the output directory
            np.save(output_file_path, npy_pc)

    print(f"All point clouds have been converted to numpy and saved to {output_dir}.")

def main_compute_normals():
    # Define the input and output directories
    input_dir = 'Data/npy_down_0.1'
    output_dir = f'Data/normals_down_0.1'

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over all .npy files in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.npy'):
            print(f"Processing {file_name}...")
            # Construct full file paths
            input_file_path = os.path.join(input_dir, file_name)
            filename_without_extension = os.path.splitext(file_name)[0]
            output_file_path = os.path.join(output_dir, filename_without_extension)

            # Perform voxel downsampling
            normals = compute_normals(input_file_path, o3d.geometry.KDTreeSearchParamKNN(30))

            # Save the downsampled point cloud to the output directory
            np.save(output_file_path, normals)

    print(f"All normals have been computed and saved to {output_dir}.")

def main_normalize(folder_path):
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Process only .npy files
        if os.path.isfile(file_path) and file_path.endswith('.npy'):
            # Load the point cloud
            point_cloud = np.load(file_path)

            # Separate RGB and other columns
            rgb_values = point_cloud[:, :3]  # Assuming the first three columns are RGB
            other_columns = point_cloud[:, 3:]

            # Normalize the RGB values
            normalized_rgb = normalize_point_cloud(rgb_values)

            # Combine back the normalized RGB with other columns
            normalized_point_cloud = np.hstack((normalized_rgb, other_columns))

            # Save the point cloud back to its original location
            np.save(file_path, normalized_point_cloud)
            print(f"Processed {filename}")

def main_normals_translation(folder_path):
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Process only .npy files
        if os.path.isfile(file_path) and file_path.endswith('.npy'):
            # Load the point cloud
            normals = np.load(file_path)
            # Transform the normals to 0-255 range
            transformed_normals = transform_normals_to_0_255(normals)
            # Save the point cloud back to its original location
            np.save(file_path, transformed_normals)
            print(f"Processed {filename}")

if __name__ == "__main__":
    # main_normals_translation("Data/normals_down_0.1")
    main_laz2npy()