
from __future__ import print_function
import os
import argparse
import time
import numpy as np
import torch
import sys
sys.path.append('./models/')
from model import PointNet_IID, PoInt_Net_only_alb  # Assuming PointNet_IID is in pointnet.model

# Define the function for image transformation
def torch2img(tensor, height, width):
    """
    Transform a torch tensor to an image.
    Args:
        tensor: PyTorch tensor to be transformed.
        height: Height of the output image.
        width: Width of the output image.
    Returns:
        NumPy array representing the image.
    """
    tensor = tensor.squeeze(0).detach().cpu().numpy()
    tensor = tensor.reshape(3, height, width)
    image = tensor.transpose(1, 2, 0) * 255
    return image

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=100, help='input batch size')
parser.add_argument('--nepoch', type=int, default=3000, help='number of epochs to train for')
parser.add_argument('--workers', type=int, default=6, help='number of data loading workers')
parser.add_argument('--lrate', type=float, default=0.0003, help='learning rate')
parser.add_argument('--env', type=str, default='mian', help='visdom environment')
parser.add_argument('--pth_path', type=str, default='')
parser.add_argument('--foldnum', type=int, default=1, help='fold number')
parser.add_argument('--sizes', type=int, default=16, help='size_scale_16_or_64')
parser.add_argument('--gpu_ids', type=str, default='0', help='choose GPU id')
parser.add_argument('--Ka', type=float, default=1.0, help='Ka for VM')
opt = parser.parse_args()

# Set CUDA visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids

# Initialize the network
PointsNet = PoInt_Net_only_alb(k=3)  # Replace with your network
network = PointsNet.cuda()
network.load_state_dict(torch.load('./pre_trained_model/all_intrinsic.pth'))
network.eval()

# Perform inference
with torch.no_grad():
    img = np.load('Data/test/final_2452_9708_re.npy')
    img = img.transpose(1, 0)
    img_tensor = torch.from_numpy(img.copy()).unsqueeze(0).float().cuda()

    # pred_alb, _, _ = network(img_tensor)
    pred_alb = network(img_tensor)
    final_img = img_tensor + 0  # Ensure a copy is made
    final_img[:, 3:, :] = pred_alb

    final_np = final_img.squeeze(0).detach().cpu().numpy().transpose(1, 0)
    np.save('Data/test/final_2452_9708_alb.npy', final_np)
    print('done!')