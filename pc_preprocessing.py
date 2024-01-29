import numpy as np
import open3d as o3d
import laspy
import os
from scipy.spatial import cKDTree
import argparse 
from plyfile import PlyData

def convert_ply_to_npy_with_i(input_folder, output_folder):
    for dirpath, dirnames, filenames in os.walk(input_folder):
        for filename in filenames:
            if filename.endswith('.ply'):
                ply_file_path = os.path.join(dirpath, filename)

                relative_path = os.path.relpath(dirpath, input_folder)
                output_subfolder = os.path.join(output_folder, relative_path)
                if not os.path.exists(output_subfolder):
                    os.makedirs(output_subfolder)

                # Read PLY file using plyfile for XYZ and intensity
                plydata = PlyData.read(ply_file_path)
                vertex = plydata['vertex']
                
                # Extract coordinates (XYZ) and intensity
                points = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
                intensity = np.array(vertex['scalar_Intensity']).reshape(-1, 1)
                combined_data = np.hstack((points, intensity))

                # Read the same PLY file using Open3D for RGB color data
                pcd = o3d.io.read_point_cloud(ply_file_path)
                if pcd.has_colors():
                    colors = np.asarray(pcd.colors)
                    combined_data = np.hstack((combined_data[:, :3], colors, combined_data[:, 3:]))

                npy_file_name = os.path.splitext(filename)[0] + ".npy"
                npy_file_path = os.path.join(output_subfolder, npy_file_name)
                np.save(npy_file_path, combined_data)
                print(f"Converted {ply_file_path} to {npy_file_path}")

def voxel_downsample_folder(input_folder, voxel_size):
    """
    Downsamples all point clouds stored in NumPy files in a given folder using voxel downsampling
    and overwrites the original files with the downsampled versions.

    Parameters:
    - input_folder: The folder containing the NumPy files with point cloud data.
    - voxel_size: The size of the voxel to downsample the point cloud.

    Returns:
    - None
    """

    # Iterate over all files in the input folder
    for file in os.listdir(input_folder):
        if file.endswith('.npy'):
            np_file_path = os.path.join(input_folder, file)

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

            # Overwrite the original file with the downsampled point cloud
            np.save(np_file_path, downsampled_point_cloud_np)
            print(f"Downsampled and overwrote {np_file_path}")

def normalize_point_clouds_in_folder(folder_path):
    """
    Normalizes the XYZ coordinates of point clouds in .npy files within a given folder.

    Parameters:
    - folder_path: The folder containing the .npy files with point cloud data.

    The function normalizes the XYZ coordinates of each point cloud by subtracting the mean 
    and dividing by the maximum absolute value, preserving the geometry of the point cloud.
    The normalized point clouds overwrite the original files.
    """

    for file in os.listdir(folder_path):
        if file.endswith('.npy'):
            file_path = os.path.join(folder_path, file)
            point_cloud = np.load(file_path)

            # Check if the file has at least three columns
            if point_cloud.shape[1] < 3:
                print(f"File {file} does not have enough columns to represent XYZ coordinates.")
                continue

            # Subtract the mean
            mean_val = np.mean(point_cloud[:, :3], axis=0)
            point_cloud[:, :3] -= mean_val

            # Divide by the maximum absolute value
            max_abs_val = np.max(np.abs(point_cloud[:, :3]))
            if max_abs_val != 0:
                point_cloud[:, :3] /= max_abs_val

            # Save the normalized point cloud back to the file
            np.save(file_path, point_cloud)
            print(f"Normalized and saved point cloud in {file_path}")

def process_and_normalize_normals(input_folder, output_folder, search_param):
    """
    Processes and normalizes the point clouds in .npy files within a given input folder.

    Parameters:
    - input_folder: The folder containing the .npy files with point cloud data.
    - output_folder: The folder where the normalized normals will be saved.
    - search_param: The search parameters for normal estimation (e.g., radius or KNN).

    The function computes normals for each point cloud, normalizes them to the range [0, 255],
    and saves these normalized normals in the output folder.
    """

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in os.listdir(input_folder):
        if file.endswith('.npy'):
            np_file_path = os.path.join(input_folder, file)
            
            # Load the point cloud data from a NumPy file
            point_cloud_np = np.load(np_file_path)

            # Convert to Open3D point cloud
            point_cloud_o3d = o3d.geometry.PointCloud()
            point_cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud_np[:, :3])

            # Estimate normals
            point_cloud_o3d.estimate_normals(search_param=search_param)

            # Convert normals to NumPy array
            normals_np = np.asarray(point_cloud_o3d.normals)

            # Normalize normals from -1 to 1 to 0 to 255
            normalized_normals = (normals_np + 1) / 2 * 255

            # Save the normalized normals to the output folder
            normalized_file_path = os.path.join(output_folder, f"{file}")
            np.save(normalized_file_path, normalized_normals)
            print(f"Processed and saved normalized normals for {np_file_path} in {output_folder}")

def main(input_folder, output_folder, output_normal):
    convert_ply_to_npy_with_i(input_folder, output_folder)
    voxel_downsample_folder(output_folder, 0.1)
    normalize_point_clouds_in_folder(output_folder)
    process_and_normalize_normals(output_folder, output_normal, o3d.geometry.KDTreeSearchParamKNN(30))

if __name__ == "__main__":
    input_folder = 'Data/ply_pc'
    output_folder = 'Data/pcd/pcd_with_i_0.1'
    output_normal = 'Data/gts/nm_0.1'
    main(input_folder, output_folder, output_normal)

