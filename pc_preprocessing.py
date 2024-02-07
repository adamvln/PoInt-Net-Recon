#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Sep 29 17:29:45 2022
@author: xxing
"""
import os
import numpy as np
import open3d as o3d
import laspy 
from scipy.spatial import cKDTree

def process_laz_files(input_folder, output_folder):
    """
    Convert all LAZ files in an input folder to NumPy arrays and save them in an output folder.

    Parameters:
    input_folder: str
        The folder path containing LAZ files.
    output_folder: str
        The folder path where converted NumPy arrays will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.laz'):
            laz_file_path = os.path.join(input_folder, filename)
            print(f'Processing {filename}...')

            # Read the .laz file
            las = laspy.read(laz_file_path)
            point_cloud_data = np.vstack((las.x, las.y, las.z)).transpose()

            # Check for additional attributes like color and intensity
            if np.all([hasattr(las, attr) for attr in ["red", "green", "blue"]]):
                colors = np.vstack((las.red, las.green, las.blue)).transpose()
                point_cloud_data = np.hstack((point_cloud_data, colors))

            if hasattr(las, 'intensity'):
                intensity = las.intensity[:, np.newaxis]
                point_cloud_data = np.hstack((point_cloud_data, intensity))

            # Save the NumPy array to the output folder
            output_filename = os.path.splitext(filename)[0] + '.npy'
            output_file_path = os.path.join(output_folder, output_filename)
            np.save(output_file_path, point_cloud_data)
            print(f'Saved NumPy array to {output_file_path}')

def process_point_clouds(folder_path, voxel_size):
    """
    Process all point cloud files in a given folder to downsample them while 
    retaining the corresponding LiDAR intensity for each downsampled point.

    Args:
        folder_path (str): Path to the folder containing point cloud files.
        voxel_size (float): The voxel size for downsampling.

    Returns:
        None
    """
    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(folder_path, filename)
            pcd_data = np.load(file_path)
            print(f'Processing {filename}, original shape: {pcd_data.shape}')

            # Normalize color and separate LiDAR data
            pcd_data[:, 3:6] = pcd_data[:, 3:6] / 65536.
            pcd_lidar = pcd_data[:, 6]

            # Create Open3D point cloud object
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcd_data[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(pcd_data[:, 3:6])

            # Downsample the point cloud
            downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

            # Extract downsampled point cloud data
            downsampled_points = np.asarray(downsampled_pcd.points)
            downsampled_colors = np.asarray(downsampled_pcd.colors)

            # Use KDTree to find the nearest original point for each downsampled point
            tree = cKDTree(pcd_data[:, :3])
            _, indices = tree.query(downsampled_points, k=1)
            downsampled_lidar = pcd_lidar[indices]

            # Combine downsampled data
            combined_data = np.concatenate((downsampled_points / 1000000, 
                                            downsampled_colors, 
                                            downsampled_lidar[:, np.newaxis]), axis=1)

            # Print and save the downsampled point cloud
            print(f'Downsampled shape: {combined_data.shape}')
            print(np.max(combined_data[:,:6]))
            output_file = os.path.join(folder_path, filename)
            np.save(output_file, combined_data)
            print(f'Saved processed data to {output_file}')

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

# Example usage
def main(input_folder, output_folder, output_normal, voxel_size = 0.2):
    process_laz_files(input_folder, output_folder)
    process_point_clouds(output_folder, voxel_size)
    process_and_normalize_normals(output_folder,output_normal, o3d.geometry.KDTreeSearchParamKNN(30))

if __name__ == "__main__":
    input_folder = 'Data/laz_pc'
    output_folder = 'Data/pcd/pcd_from_laz_with_i_0.2'
    output_normal = 'Data/gts/nm_from_laz_0.2'
    main(input_folder, output_folder, output_normal)
