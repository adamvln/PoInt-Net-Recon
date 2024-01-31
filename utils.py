import numpy as np

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
    reconstructed = np.clip(reconstructed, 0, 1)
    return reconstructed