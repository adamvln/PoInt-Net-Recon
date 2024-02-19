import os
import argparse
import torch
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import sys
import wandb

torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
torch.cuda.empty_cache()
# sys.path.append('./models/')

from utils import *
from models.model import *
from models.dataset import *

def torch2img(tensor, h, w):
    """
    Convert a PyTorch tensor to a numpy image.

    Parameters:
    tensor (torch.Tensor): The input tensor.
    h (int): The height of the output image.
    w (int): The width of the output image.

    Returns:
    np.ndarray: The resulting image in numpy array format.
    """
    tensor = tensor.squeeze(0).detach().cpu().numpy()
    tensor = tensor.reshape(3, h, w)
    img = tensor.transpose(1, 2, 0)
    img = img * 255.
    return img

def smoothness_loss(pred_alb, n_neighbors):
    """
    Calculate the smoothness loss for the predicted albedo.

    Parameters:
    pred_alb (torch.Tensor): Predicted albedo tensor.
    n_neighbors (int): Number of neighbors to consider for each point.

    Returns:
    torch.Tensor: Calculated smoothness loss.
    """
    # Ensure pred_alb is detached and moved to CPU for KNN
    pred_alb_np = pred_alb.detach().cpu().numpy()

    # Apply KNN to find n_neighbors for each point
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(pred_alb_np)
    distances, indices = neigh.kneighbors(pred_alb_np)

    # Compute smoothness loss
    loss = 0.0
    for i in range(pred_alb_np.shape[0]):
        for j in indices[i]:
            loss += np.linalg.norm(pred_alb_np[i] - pred_alb_np[j])**2

    # Normalize the smoothness loss by the number of points
    loss /= pred_alb_np.shape[0]

    # Convert the loss back to a torch tensor and return
    return torch.tensor(loss, dtype=torch.float32, device=pred_alb.device)

def train_model(network, train_loader, optimizer, criterion, epochs, s1, s2,
         b1, b2, include_loss_recon = True, include_loss_lid=False, include_loss_smoothness = False, loss_lid_coeff=1.0, smoothness_loss_coeff = 1.0):
    """
    Train the PoIntNet model.

    Parameters:
    network (torch.nn.Module): The neural network model.
    train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
    optimizer (torch.optim.Optimizer): Optimizer for the training.
    criterion (torch.nn.Module): Loss function.
    epochs (int): Number of epochs to train.
    include_loss_lid (bool): Flag to include loss_lid in the total loss computation.
    loss_lid_coeff (float): Coefficient to scale the impact of loss_lid.

    Returns:
    None
    """
    network.train()
    print("The lidar loss coefficient is {}".format(loss_lid_coeff))
    print("The smoothness loss coefficient is {}".format(smoothness_loss_coeff))
    for epoch in range(epochs):
        running_loss = 0.0
        for data in tqdm(train_loader):
            # Load the data and transfer to GPU
            img, norms, lid, fn = data
            img, norms, lid = img.cuda(), norms.cuda(), lid.cuda()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            pred_shd, pred_alb = network(img, norms, point_pos_in=1, ShaderOnly=False)
            gray_alb, gray_shd = point_cloud_to_grayscale_torch(pred_alb), point_cloud_to_grayscale_torch(pred_shd)

            # Reconstruct the point cloud
            reconstructed_pcd = reconstruct_image(pred_alb, pred_shd)
            loss = 0
            # Compute loss
            if include_loss_recon:
                loss += criterion(reconstructed_pcd, img[:,3:6])
            
            if include_loss_lid:
                epsilon = 1e-8
                lid_normalized = lid / 65535.0
                loss_lid = torch.abs(gray_alb - s1 * lid_normalized - b1) + torch.abs(gray_shd - s2 * (gray_alb/(lid_normalized + epsilon)) - b2)
                loss += loss_lid_coeff * loss_lid.mean()
            
            if include_loss_smoothness:
                loss += smoothness_loss_coeff * smoothness_loss(pred_alb, smoothness_loss_coeff)
                
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Logging (e.g., with wandb)
            wandb.log({"batch_loss": loss.item()})

        epoch_loss = running_loss / len(train_loader)
        wandb.log({"epoch_loss": epoch_loss, "epoch": epoch})
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

def setup_network(model_path):
    """
    Set up and return the PointNet network.

    Returns:
    torch.nn.Module: The initialized PointNet network.
    """
    PoIntNet = PoInt_Net(k=3)
    network = PoIntNet.cuda()

    # Load the entire checkpoint
    checkpoint = torch.load(model_path)

    # Extract only the model's state_dict
    if 'model_state_dict' in checkpoint:
        network.load_state_dict(checkpoint['model_state_dict'])
    else:
        # If the model_state_dict key is not present, assume the entire file is the state_dict
        network.load_state_dict(checkpoint)

    return network

def main_train():
    """
    Main training function. Parses command-line arguments, initializes the model,
    dataloader, loss function, and optimizer, then starts the training process.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')

    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
    parser.add_argument('--gpu_ids', type=str, default='0', help='choose GPU')
    
    parser.add_argument('--path_to_pc', type=str, default='./Data/pcd/pcd_from_laz_with_i_0.3/', help='path to test data')
    parser.add_argument('--path_to_nm', type=str, default='./Data/gts/nm_from_laz_0.3/', help='path to test data')
    parser.add_argument('--save_model_path', type=str, default='./pre_trained_model/only_lid_loss_{lr:.4f}.pth', help='path to save the trained model')
    
    parser.add_argument('--include_loss_recon', type=bool, default=True, help='whether to include reconstruction loss in the total loss computation')
    parser.add_argument('--include_loss_smoothness', type=bool, default=False, help='whether to include smoothness loss in the total loss computation')
    parser.add_argument('--include_loss_lid', type=bool, default=False, help='whether to include loss_lid in the total loss computation')
    
    parser.add_argument('--loss_lid_coeff', type=float, default=1.0, help='coefficient to scale the impact of loss_lid')
    parser.add_argument('--loss_smoothness_coeff', type=float, default=1.0, help='coefficient to scale the impact of smoothness loss')
    opt = parser.parse_args()

    # Format the save_model_path with the learning rate
    opt.save_model_path = opt.save_model_path.format(lr=opt.lr)

    # Initialize wandb
    wandb.init(project="iid_pc", config={
        "epochs": opt.epochs,
        "batch_size": opt.batch_size,
        "learning_rate": opt.lr,
        "loss_lid_coeff" : opt.loss_lid_coeff,
        "s1_init": 1.0,
        "s2_init": 1.0,
        "b1_init": 0.0,
        "b2_init": 0.0,
    })

    wandb.init(project="iid_pc", name=f"lr_{wandb.config.learning_rate}_batch_{wandb.config.batch_size}")
    config = wandb.config
    # Set the GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids

    # Load dataset and create DataLoader
    dataset = PcdIID_Recon(opt.path_to_pc, opt.path_to_nm, 
                           train=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, 
                                             shuffle=False, num_workers=opt.workers, 
                                             collate_fn=custom_collate_fn, drop_last=True)

    # Initialize the network
    network = setup_network('./pre_trained_model/all_intrinsic.pth')
    s1 = nn.Parameter(torch.tensor([wandb.config.s1_init], device='cuda'))
    s2 = nn.Parameter(torch.tensor([wandb.config.s2_init], device='cuda'))
    b1 = nn.Parameter(torch.tensor([wandb.config.b1_init], device='cuda'))
    b2 = nn.Parameter(torch.tensor([wandb.config.b2_init], device='cuda'))
    # network = network.half()

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD([
        {'params': network.parameters()},
        {'params': [s1, s2, b1, b2]}
    ], lr=opt.lr)

    # Start the training
    train_model(network, dataloader, optimizer, criterion, opt.epochs, s1, s2, b1, b2, opt.include_loss_lid, opt.loss_lid_coeff)

    # Save the trained model
    torch.save({
        'model_state_dict': network.state_dict(),
        's1': s1.item(),
        's2': s2.item(),
        'b1': b1.item(),
        'b2': b2.item()
    }, opt.save_model_path)

    wandb.save(opt.save_model_path)
    wandb.finish()

if __name__ == "__main__":
    main_train()
