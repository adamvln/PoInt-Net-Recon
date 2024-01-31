"""
All copyright resevered!
Only for reviewing

DO NOT SHARE!

Author of paper 2847
"""

from __future__ import print_function
import os
import sys
import argparse
import random
import json
import time
import datetime
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable

sys.path.append('./models/')
from model import *
from dataset import *

def torch2img(tensor,h,w):
    tensor = tensor.squeeze(0).detach().cpu().numpy()
    tensor = tensor.reshape(3,h,w)
    img = tensor.transpose(1,2,0)
    img = img*255.
    return img



parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--gpu_ids', type=str, default='0', help='chose GPU')
opt = parser.parse_args()
print(opt)
log_name = 'IID'+'_others_'
log_path = os.path.join('./test_results/',log_name)
if not os.path.exists(log_path):
    os.makedirs(log_path)
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
# load data
# dataset_test = PcdIID(train=False)
dataset_test = PcdIID_Recon('./Data/pcd/pcd_from_laz_with_i_0.1/', './Data/gts/nm_from_laz_0.1/', train=False)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=opt.workers)
len_dataset_test = len(dataset_test)
print('len_dataset_test:', len(dataset_test))

# Initialize the PointNet network with k=3
PoIntNet = PoInt_Net(k=3) 

# Move the network to the GPU for faster computation
network = PoIntNet.cuda()

# Load the pretrained model weights into the network
network.load_state_dict(torch.load('./pre_trained_model/all_intrinsic.pth'))
print('start train.....')

# Disable gradient computations for evaluation (saves memory and computations)
with torch.no_grad():
    # Set the network to evaluation mode (affects layers like dropout and batchnorm)
    network.eval()

    # Record the start time for evaluating performance
    start = time.time()

    # Iterate over the test dataset
    for i, data in tqdm(enumerate(dataloader_test)):
        # Unpack the data (image, normals, filename)
        img, norms,_, fn = data
        print(img.shape)
        # Move the data to the GPU
        img = img.cuda()
        norms = norms.cuda()

        # Forward pass through the network to get predictions
        pred_shd, pred_alb = network(img, norms, point_pos_in=1, ShaderOnly=False)
        final_alb = img + 0  # Ensure a copy is made
        final_alb[:, 3:, :] = pred_alb

        final_shd = img + 0
        final_shd[:, 3:, :] = pred_shd
        
        # # Save the predicted albedo and shading numpy arrays
        np.save(log_path + '/albedo_estimate/' + str(fn[0]) + '_alb.npy', final_alb.squeeze(0).transpose(1,0).detach().cpu().numpy())
        np.save(log_path + '/shading_estimate/' + str(fn[0]) + '_shd.npy', final_shd.squeeze(0).transpose(1,0).detach().cpu().numpy())

        # # Set image dimensions
        # w, h = 512, 512

        # # Get the batch size from the predicted albedo shape
        # n_b = pred_alb.shape[0]

        # # Extract RGB channels from the image tensor
        # img_tensor = img[:, 3:, :]

        # # Convert tensors to images for visualization
        # img_pic = torch2img(img_tensor, h, w)
        # alb_pic = torch2img(pred_alb, h, w)
        # shd_pic = torch2img(pred_shd, h, w)

        # # Concatenate the original, albedo, and shading images horizontally
        # img_final = np.concatenate((img_pic, alb_pic, shd_pic), axis=1) 

        # # Save the concatenated image. Reverse channels from BGR to RGB for correct color display
        # cv2.imwrite(log_path + '/' + str(fn[0]) + 'all.png', img_final[..., ::-1])

    # Calculate and print the total time taken for the evaluation
    time_use2 = time.time() - start
    print(time_use2)

