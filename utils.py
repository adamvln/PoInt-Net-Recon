import numpy as np
import torch 
import re 
import open3d as o3d
from tqdm import tqdm 
import os 

def reconstruct_image(albedo, shading):
    """
    Reconstruct an image by element-wise multiplication of albedo and shading images.

    Parameters:
    albedo (np.ndarray): Albedo image.
    shading (np.ndarray): Shading image.

    Returns:
    np.ndarray: Reconstructed image.
    """
    reconstructed = albedo * shading
    # Assuming the images are normalized (0 to 1), if they are not, adjust accordingly
    reconstructed = torch.clamp(reconstructed, 0.00001, 1)
    return reconstructed

def point_cloud_to_grayscale_torch(rgb_values):

        """
        Convert RGB values of a point cloud to grayscale using PyTorch for a batched input.

        Parameters:
        rgb_values (torch.Tensor): A tensor of shape (batch_size, 3, N) where N is the number of points.
                                   Each point is represented as (r, g, b) across the batch.

        Returns:
        torch.Tensor: Grayscale values of the point cloud of shape (batch_size, 1, N).
        """
        # Ensure input is a PyTorch tensor and has the correct shape (3D tensor for batched input)
        if not isinstance(rgb_values, torch.Tensor):
            rgb_values = torch.tensor(rgb_values, dtype=torch.float32)

        # Ensure the tensor is on the correct device (e.g., CUDA)
        rgb_values = rgb_values.to("cuda")

        # Check if the input tensor is 3D
        if rgb_values.dim() != 3:
            raise ValueError("Input tensor should be a 3D tensor with shape [batch_size, channels, points]")
        
        # Define weights for luminosity method and adjust for batch operation
        weights = torch.tensor([0.21, 0.72, 0.07], dtype=torch.float32, device=rgb_values.device)
        weights = weights.view(1, 3, 1)  # Reshape for broadcasting across the batch and points
        # Calculate grayscale values using luminosity method
        grayscale_values = torch.mul(rgb_values, weights).sum(dim=1, keepdim=True)
        # Squeeze the singleton dimension to match expected output shape
        grayscale_values = grayscale_values.squeeze(1)
        return grayscale_values

def pad_point_cloud(tensor, max_points):
    """
    Pads the point cloud tensor to a specified maximum number of points.

    Parameters:
    tensor (torch.Tensor): The point cloud tensor.
    max_points (int): The maximum number of points to pad to.

    Returns:
    torch.Tensor: Padded point cloud tensor.
    """
    num_points = tensor.shape[1]
    
    if num_points < max_points:
        # Padding required
        padding_size = max_points - num_points
        padding = torch.zeros((tensor.shape[0], padding_size), dtype=tensor.dtype, device=tensor.device)
        tensor = torch.cat([tensor, padding], dim=1)
    return tensor

def custom_collate_fn(batch):
    """
    Custom collate function to pad point clouds in the batch to the same size.

    Parameters:
    batch (list): A list of tuples with point cloud data.

    Returns:
    list: Collated batch with padded point clouds.
    """
    # Find the maximum number of points in any point cloud in the batch
    max_points = max(pc.shape[1] for _, pc, _, _ in batch)

    # Apply padding to each point cloud in the batch
    new_batch = []
    for data in batch:
        img, norms, lid, fn = data
        img = pad_point_cloud(img, max_points)
        norms = pad_point_cloud(norms, max_points)
        lid = pad_point_cloud(lid, max_points)  # Assuming 'lid' is also a point cloud tensor
        new_batch.append((img, norms, lid, fn))
    
    # Default collate should work now
    return torch.utils.data.dataloader.default_collate(new_batch)

def detect_edge_points(point_cloud_file_path, save_path, save_mask=True):
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

    # Convert numpy array to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_np)

    # Compute the normals of the point cloud
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Create a KDTree for the point cloud
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    # Detect edge points based on the variation of normals
    normals = np.asarray(pcd.normals)
    edge_indices = []
    threshold = 0.9  # Threshold for edge detection, adjust as needed

    for i in tqdm(range(len(normals))):
        k, idx, _ = pcd_tree.search_radius_vector_3d(pcd.points[i], 0.1)
        local_normals = normals[idx[1:], :]
        cos_similarity = np.dot(local_normals, normals[i])

        if np.any(cos_similarity < threshold):
            edge_indices.append(i)

    # Create a binary mask
    binary_mask = np.zeros(len(pcd.points), dtype=int)
    binary_mask[edge_indices] = 1

    if save_mask:
        # Extract point cloud number from file name
        pc_number = os.path.basename(point_cloud_file_path).split('.')[0].split('_')[1:]
        pc_number = '_'.join(pc_number)
        voxel_size = re.findall(r"\d+\.\d+", input_string)[0] if re.findall(r"\d+\.\d+", input_string) else None
        mask_filename = f'final_{pc_number}_{voxel_size}_edge_mask.npy'
        mask_filepath = os.path.join(save_path, mask_filename)
        np.save(mask_filepath, binary_mask)

    return binary_mask

detect_edge_points("Data/pcd/pcd_from_laz_with_i_0.3/final_2448_9707.npy", "Data/edge_masks")