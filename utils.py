import numpy as np
import torch 
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

def compute_luminance_and_chromaticity_batched(point_clouds):
    # Assume point_clouds is a tensor of shape (B, 6, N) where B is the batch size and the second dimension contains XYZRGB
    rgb = point_clouds[:, 3:, :]  # Extract the RGB values, assuming they are the last three channels
    # Normalize RGB values to [0, 1] if they are in [0, 255]
    if torch.max(rgb) > 1.0:
        rgb = rgb / 255.0

    # Compute luminance
    luminance = 0.2126 * rgb[:, 0, :] + 0.7152 * rgb[:, 1, :] + 0.0722 * rgb[:, 2, :]
    
    # Compute chromaticity
    chroma_sum = torch.sum(rgb, dim=1, keepdim=True)
    chromaticity_x = rgb[:, 0, :] / (chroma_sum + 1e-8).squeeze(1)  # Squeeze to remove the singleton dimension
    chromaticity_y = rgb[:, 1, :] / (chroma_sum + 1e-8).squeeze(1)
    
    # Stack the chromaticity coordinates for each point across the batch
    chromaticity = torch.stack((chromaticity_x, chromaticity_y), dim=1)
    
    # The shape of luminance will be (B, N)
    # The shape of chromaticity will be (B, 2, N)
    return luminance, chromaticity

def compute_weights(ch_i, ch_j, lum_i, lum_j):
    # Compute the squared chromaticity difference for x and y, and then sum them
    chromaticity_difference_squared = torch.sum((ch_i - ch_j) ** 2)

    # Compute the chromaticity difference
    chromaticity_difference = torch.sqrt(chromaticity_difference_squared)

    # Compute the weight using the provided formula
    # Note: No need to calculate max difference since it's for individual points
    weight = (1 - chromaticity_difference) * torch.sqrt(lum_i * lum_j)
    
    return weight

def compute_weights_vectorized(ch_i, ch_j, lum_i, lum_j):
    """
    Compute the weights based on chromaticity and luminance differences using vectorized operations.

    Parameters:
    ch_i (torch.Tensor): Chromaticity of points i with shape [2, N, n_neighbors].
    ch_j (torch.Tensor): Chromaticity of points j with the same shape as ch_i.
    lum_i (torch.Tensor): Luminance of points i with shape [N, n_neighbors].
    lum_j (torch.Tensor): Luminance of points j with the same shape as lum_i.

    Returns:
    torch.Tensor: Weights for each pair of points with shape [N, n_neighbors].
    """
    # Compute the squared chromaticity differences for x and y, and then sum them
    chromaticity_difference_squared = torch.sum((ch_i - ch_j) ** 2, dim=0)

    # Compute the chromaticity difference
    chromaticity_difference = torch.sqrt(chromaticity_difference_squared)

    # Find the maximum chromaticity difference for normalization
    max_chromaticity_difference = chromaticity_difference.max(dim=1, keepdim=True).values

    # Compute the weights using the provided formula, avoiding division by zero
    weights = (1 - chromaticity_difference / (max_chromaticity_difference + 1e-8)) * torch.sqrt(lum_i * lum_j)

    return weights