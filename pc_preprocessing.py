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
import re 
import shutil
import random

def split_point_clouds_in_folder(input_folder, output_folder):
    # Iterate over all files in the input folder
    for file in os.listdir(input_folder):
        # Check if the file is a NumPy file
        if file.endswith('.npy'):
            file_path = os.path.join(input_folder, file)

            # Load the point cloud from a NumPy file
            point_cloud = np.load(file_path)
            # Extract file name and extension
            base_name = os.path.basename(file_path)
            name, ext = os.path.splitext(base_name)

            # Parse the original coordinates from the file name
            coords = name.split('_')
            xg, yg = int(coords[1]), int(coords[2])
            # Calculate new dimensions (half of the original)
            half_x, half_y = 25, 25

            # Iterate and create 4 new point clouds
            for i in range(2):
                for j in range(2):
                    # Calculate new bottom left corner
                    new_xg = xg * 50 + i * half_x
                    new_yg = yg * 50 + j * half_y
                    # Create new file name
                    new_file_name = f"final_{new_xg}_{new_yg}.npy"

                    # Filter the original point cloud to create a new one
                    x_min = new_xg * 50
                    x_max = x_min + half_x * 50
                    y_min = new_yg * 50
                    y_max = y_min + half_y * 50
                    mask = (
                        (point_cloud[:, 0] >= new_xg ) & (point_cloud[:, 0] < new_xg + 25) &
                        (point_cloud[:, 1] >= new_yg ) & (point_cloud[:, 1] < new_yg + 25)
                    )
                    new_point_cloud = point_cloud[mask]
                    # Save the new point cloud
                    new_file_path = os.path.join(output_folder, new_file_name)
                    np.save(new_file_path, new_point_cloud)
                    

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

def detect_edge_points(point_cloud_file_path, normals_file_path, save_path, save_mask=True):
    """
    Detects edge points in a point cloud and saves the binary mask.

    Parameters:
    point_cloud_file_path (str): Path to the numpy file containing the point cloud.
    save_path (str): Directory where the binary mask will be saved.
    voxel_size (float): The downsampling voxel size.
    save_mask (bool): Whether to save the binary mask to a file.

    Returns:
    np.array: Binary mask indicating edge points.
    """
    # Load point cloud data from the numpy file
    point_cloud_np = np.load(point_cloud_file_path)
    point_cloud_np = point_cloud_np[:, :3].astype(np.float64)

    normals_np = np.load(normals_file_path)
    # Convert numpy array to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_np)

    # Compute the normals of the point cloud
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Create a KDTree for the point cloud
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    # Detect edge points based on the variation of normals
    normals = np.asarray(pcd.normals)
    edge_indices_0 = []
    # edge_indices_1 = []
    # edge_indices_2 = []
    # edge_indices_3 = []
    # edge_indices_4 = []
    # edge_indices_5 = []
    edge_indices_m1 = []
    # edge_indices_m2 = []
    # edge_indices_m3 = []
    # edge_indices_m4 = []
    # edge_indices_m5 = []
    # threshold = 0.4  # Threshold for edge detection, adjust as needed

    for i in range(len(normals)):
        k, idx, _ = pcd_tree.search_radius_vector_3d(pcd.points[i], 0.1)
        local_normals = normals[idx[1:], :]
        cos_similarity = np.dot(local_normals, normals[i])
        #Approcach with mean of cos_similarity, so if the mean of the neighbors normal vectors have a cosine simi
        #larity less than the threshold, then the point is an edge point	
        mean_cos_similarity = np.mean(cos_similarity)
        #[ 0.99981499  0.99885637  0.99905021 ... -0.27847014 -0.27844502 -0.27844502]
        if mean_cos_similarity < 0:
            edge_indices_0.append(i)
        # if mean_cos_similarity < 0.1:
        #     edge_indices_1.append(i)
        # if mean_cos_similarity < 0.2:
        #     edge_indices_2.append(i)
        # if mean_cos_similarity < 0.3:
        #     edge_indices_3.append(i)
        # if mean_cos_similarity < 0.4:
        #     edge_indices_4.append(i)
        # if mean_cos_similarity < 0.5:
        #     edge_indices_5.append(i)
        if mean_cos_similarity < -0.1:
            edge_indices_m1.append(i)
        # if mean_cos_similarity < -0.2:
        #     edge_indices_m2.append(i)
        # if mean_cos_similarity < -0.3:
        #     edge_indices_m3.append(i)
        # if mean_cos_similarity < -0.4:
        #     edge_indices_m4.append(i)
        # if mean_cos_similarity < -0.5:
        #     edge_indices_m5.append(i)

    # Create a binary mask
    binary_mask_0 = np.zeros(len(pcd.points), dtype=int)
    # binary_mask_1 = np.zeros(len(pcd.points), dtype=int)
    # binary_mask_2 = np.zeros(len(pcd.points), dtype=int)
    # binary_mask_3 = np.zeros(len(pcd.points), dtype=int)
    # binary_mask_4 = np.zeros(len(pcd.points), dtype=int)
    # binary_mask_5 = np.zeros(len(pcd.points), dtype=int)
    binary_mask_m1 = np.zeros(len(pcd.points), dtype=int)
    # binary_mask_m2 = np.zeros(len(pcd.points), dtype=int)
    # binary_mask_m3 = np.zeros(len(pcd.points), dtype=int)
    # binary_mask_m4 = np.zeros(len(pcd.points), dtype=int)
    # binary_mask_m5 = np.zeros(len(pcd.points), dtype=int)

    binary_mask_0[edge_indices_0] = 1
    # binary_mask_1[edge_indices_1] = 1
    # binary_mask_2[edge_indices_2] = 1
    # binary_mask_3[edge_indices_3] = 1
    # binary_mask_4[edge_indices_4] = 1
    # binary_mask_5[edge_indices_5] = 1
    binary_mask_m1[edge_indices_m1] = 1
    # binary_mask_m2[edge_indices_m2] = 1
    # binary_mask_m3[edge_indices_m3] = 1
    # binary_mask_m4[edge_indices_m4] = 1
    # binary_mask_m5[edge_indices_m5] = 1
    
    if save_mask:
        # Extract point cloud number from file name
        pc_number = os.path.basename(point_cloud_file_path).split('.')[0].split('_')[1:]
        pc_number = '_'.join(pc_number)
        voxel_size = re.findall(r"\d+\.\d+", point_cloud_file_path)[0] if re.findall(r"\d+\.\d+", point_cloud_file_path) else None
        mask_filename_0 = f'final_{pc_number}_mean_{0}.npy'
        # mask_filename_1 = f'final_{pc_number}_{voxel_size}_edge_mask_threshold_mean_{0.1}.npy'
        # mask_filename_2 = f'final_{pc_number}_{voxel_size}_edge_mask_threshold_mean_{0.2}.npy'
        # mask_filename_3 = f'final_{pc_number}_{voxel_size}_edge_mask_threshold_mean_{0.3}.npy'
        # mask_filename_4 = f'final_{pc_number}_{voxel_size}_edge_mask_threshold_mean_{0.4}.npy'
        # mask_filename_5 = f'final_{pc_number}_{voxel_size}_edge_mask_threshold_mean_{0.5}.npy'
        mask_filename_m1 = f'final_{pc_number}_mean_{-0.1}.npy'
        # mask_filename_m2 = f'final_{pc_number}_{voxel_size}_edge_mask_threshold_mean_{-0.2}.npy'
        # mask_filename_m3 = f'final_{pc_number}_{voxel_size}_edge_mask_threshold_mean_{-0.3}.npy'
        # mask_filename_m4 = f'final_{pc_number}_{voxel_size}_edge_mask_threshold_mean_{-0.4}.npy'
        # mask_filename_m5 = f'final_{pc_number}_{voxel_size}_edge_mask_threshold_mean_{-0.5}.npy'

        mask_filepath_0 = os.path.join(save_path, mask_filename_0)
        # mask_filepath_1 = os.path.join(save_path, mask_filename_1)
        # mask_filepath_2 = os.path.join(save_path, mask_filename_2)
        # mask_filepath_3 = os.path.join(save_path, mask_filename_3)
        # mask_filepath_4 = os.path.join(save_path, mask_filename_4)
        # mask_filepath_5 = os.path.join(save_path, mask_filename_5)
        mask_filepath_m1 = os.path.join(save_path, mask_filename_m1)
        # mask_filepath_m2 = os.path.join(save_path, mask_filename_m2)
        # mask_filepath_m3 = os.path.join(save_path, mask_filename_m3)
        # mask_filepath_m4 = os.path.join(save_path, mask_filename_m4)
        # mask_filepath_m5 = os.path.join(save_path, mask_filename_m5)

        np.save(mask_filepath_0, binary_mask_0)
        # np.save(mask_filepath_1, binary_mask_1)
        # np.save(mask_filepath_2, binary_mask_2)
        # np.save(mask_filepath_3, binary_mask_3)
        # np.save(mask_filepath_4, binary_mask_4)
        # np.save(mask_filepath_5, binary_mask_5)
        np.save(mask_filepath_m1, binary_mask_m1)
        # np.save(mask_filepath_m2, binary_mask_m2)
        # np.save(mask_filepath_m3, binary_mask_m3)
        # np.save(mask_filepath_m4, binary_mask_m4)
        # np.save(mask_filepath_m5, binary_mask_m5)


def main(input_folder, output_folder, split_folder, output_normal, voxel_size = 0.3):
    process_laz_files(input_folder, output_folder)
    split_point_clouds_in_folder(output_folder, split_folder)
    process_point_clouds(split_folder, voxel_size)
    process_and_normalize_normals(split_folder,output_normal, o3d.geometry.KDTreeSearchParamKNN(30))

def move_files(src_folder, dest_folder, files):
    for file in files:
        shutil.move(os.path.join(src_folder, file), os.path.join(dest_folder, file))

if __name__ == "__main__":
    # input_folder = 'Data/laz_pc'
    # output_folder = 'Data/pcd/pcd_0.3'
    # split_folder = 'Data/pcd/pcd_split_0.3'
    # output_normal = 'Data/gts/nm_split_0.3'
    # main(input_folder, output_folder, split_folder, output_normal)
    # # input_folder = 'Data/pcd/pcd_from_laz_with_i_0.1'
    # # normal_folder = 'Data/gts/nm_from_laz_0.1'
    # for file in os.listdir(split_folder):
    #     file_path = os.path.join(split_folder, file)
    #     normal_path = os.path.join(output_normal, file)
    #     detect_edge_points(file_path, normal_path, "Data/edge_masks/edge_masks_0.3")
        # break
        # detect_edge_points("Data/pcd/pcd_from_laz_with_i_0.1/final_2448_9707.npy", "Data/gts/nm_from_laz_0.1/final_2448_9707.npy", "Data/edge_masks")

        # Define the folders
