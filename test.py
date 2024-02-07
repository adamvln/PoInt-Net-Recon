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

def test_model(network, dataloader, log_path):
    """
    Perform inference on a dataset using a given network and save the output.

    Parameters:
    network (torch.nn.Module): The neural network model for inference.
    dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
    log_path (str): Path to save the inference results.

    Returns:
    None
    """
    with torch.no_grad():
        network.eval()
        start = time.time()

        for i, data in tqdm(enumerate(dataloader)):
            img, norms, _, fn = data
            img = img.cuda()
            norms = norms.cuda()

            print(img.shape, norms.shape)

            pred_shd, pred_alb = network(img, norms, point_pos_in=1, ShaderOnly=False)

            final_alb = img + 0
            final_alb[:, 3:, :] = pred_alb
            final_shd = img + 0
            final_shd[:, 3:, :] = pred_shd

            np.save(os.path.join(log_path, 'albedo_estimate', f"{fn[0]}_alb.npy"), 
                    final_alb.squeeze(0).transpose(1, 0).detach().cpu().numpy())
            np.save(os.path.join(log_path, 'shading_estimate', f"{fn[0]}_shd.npy"), 
                    final_shd.squeeze(0).transpose(1, 0).detach().cpu().numpy())

        time_use = time.time() - start
        print(f"Total time for evaluation: {time_use} seconds")

def train_model(network, train_loader, optimizer, criterion, epochs):
    """
    Train the PoIntNet model.

    Parameters:
    network (torch.nn.Module): The neural network model.
    train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
    optimizer (torch.optim.Optimizer): Optimizer for the training.
    criterion (torch.nn.Module): Loss function.
    epochs (int): Number of epochs to train.

    Returns:
    None
    """
    network.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for data in tqdm(train_loader):
            # Load the data and transfer to GPU
            img, norms, lid, fn = data
            img, norms, lid = img.cuda(), norms.cuda(), lid.cuda()
            img, norms, lid = img.half(), norms.half(), lid.half()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            pred_shd, pred_alb = network(img, norms, point_pos_in=1, ShaderOnly=False)

            # Reconstruct the point cloud
            reconstructed_pcd = reconstruct_image(pred_alb, pred_shd)
            # Compute loss
            # img_color  = torch.clamp(img[:,3:6], 0.00001, 1)
            loss = criterion(reconstructed_pcd, img[:,3:6]) 
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            wandb.log({"batch_loss": loss.item()})

        epoch_loss = running_loss / len(train_loader)
        wandb.log({"epoch_loss": epoch_loss, "epoch": epoch})
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

def setup_network():
    """
    Set up and return the PointNet network.

    Returns:
    torch.nn.Module: The initialized PointNet network.
    """
    PoIntNet = PoInt_Net(k=3)
    network = PoIntNet.cuda()
    network.load_state_dict(torch.load('./pre_trained_model/all_intrinsic.pth'))
    return network

def main_test():
    """
    Main function to handle argument parsing and initiate the inference process.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
    parser.add_argument('--gpu_ids', type=str, default='0', help='choose GPU')
    parser.add_argument('--path_to_test_pc', type=str, default='./Data/pcd/pcd_from_laz_with_i_0.1/', help='path to test data')
    parser.add_argument('--path_to_test_nm', type=str, default='./Data/gts/nm_from_laz_0.1/', help='path to test data')
    opt = parser.parse_args()

    log_name = 'IID_others_'
    log_path = os.path.join('./test_results/', log_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids

    dataset_test = PcdIID_Recon(opt.path_to_test_pc, opt.path_to_test_nm, train=False)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=opt.workers)

    print('len_dataset_test:', len(dataset_test))
    network = setup_network()
    print('Infering.....')

    test_model(network, dataloader_test, log_path)

def main_train():
    """
    Main training function. Parses command-line arguments, initializes the model,
    dataloader, loss function, and optimizer, then starts the training process.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
    parser.add_argument('--gpu_ids', type=str, default='0', help='choose GPU')
    parser.add_argument('--path_to_pc', type=str, default='./Data/pcd/pcd_from_laz_with_i_0.2/', help='path to test data')
    parser.add_argument('--path_to_nm', type=str, default='./Data/gts/nm_from_laz_0.2/', help='path to test data')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--save_model_path', type=str, default='./pre_trained_model/ft_intrinsic.pth', help='path to save the trained model')
    opt = parser.parse_args()

    # Initialize wandb
    wandb.init(project="iid_pc", config={
        "epochs": opt.epochs,
        "batch_size": opt.batch_size,
        "learning_rate": opt.lr,
        # Add other hyperparameters or configuration details here
    })
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
    network = setup_network()
    network = network.half()

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(network.parameters(), lr=opt.lr)

    # Start the training
    train_model(network, dataloader, optimizer, criterion, opt.epochs)

    # Save the trained model
    torch.save(network.state_dict(), opt.save_model_path)
    wandb.save(save_model_path)

if __name__ == "__main__":
    main_train()
