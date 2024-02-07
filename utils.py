import numpy as np
import torch 
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