import os
import argparse
import torch
import numpy as np
import torch.utils.data
from tqdm import tqdm
import time
import sys

sys.path.append('./models/')
from model import *
from dataset import *

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

def infer(network, dataloader, log_path):
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

def main():
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

    infer(network, dataloader_test, log_path)

if __name__ == "__main__":
    main()
