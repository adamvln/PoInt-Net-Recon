from __future__ import print_function
import os
import cv2
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import torch
import torch.utils.data as data




class PcdIID(data.Dataset):
    def __init__(self, train=False, foldn=0,sizes=16):
        self.file_list = os.listdir('./Data/pcd/pcd-ori/')
        self.train = train

    def __getitem__(self, index):
        fn = self.file_list[index]
        fnn = fn.strip().split('.')[0]
        pcd = np.load('./Data/pcd/pcd-ori/'+fnn+'.npy')
        norms = np.load('./Data/gts/normal-ori/'+ fnn + '.npy')
            
        p,c_p = pcd.shape
    
        pcd = np.array(pcd, dtype='float32')
        

        norms = np.array(norms,'float32')
        pcd = pcd.transpose(1,0)
        norms = norms.transpose(1,0)
        norms = torch.from_numpy(norms.copy())
        pcd = torch.from_numpy(pcd.copy())
        return pcd,norms,fnn


    def __len__(self):
        return (len(self.file_list))

class PcdIID_Recon(data.Dataset):
    def __init__(self, pcd_path, normals_path, train=False, foldn=0, sizes=16):
        """
        Initializes the PcdIID dataset class.

        Args:
            pcd_path (str): Path to the directory containing point cloud data files.
            normals_path (str): Path to the directory containing normals data files.
            train (bool, optional): Flag to indicate if the dataset is used for training. Defaults to False.
            foldn (int, optional): Fold number for cross-validation, if applicable. Defaults to 0.
            sizes (int, optional): The size of the data to be used. Defaults to 16.
        """
        self.pcd_path = pcd_path
        self.normals_path = normals_path
        self.file_list = os.listdir(pcd_path)
        self.train = train

    def __getitem__(self, index):
        """
        Retrieves a point cloud and its corresponding normals from the dataset at the specified index.

        Processing steps:
        1. Load the point cloud and normal data from their respective numpy files.
        2. Ensure the data is in the correct format (float32).
        3. Transpose the data to match the expected input format for subsequent processing.
        4. Convert numpy arrays to PyTorch tensors for compatibility with PyTorch models.

        Args:
            index (int): The index of the data item to be retrieved.

        Returns:
            tuple: A tuple containing the point cloud tensor, normals tensor, and filename without extension.
        """
        fn = self.file_list[index]
        fnn = fn.strip().split('.')[0]
        pcd = np.load(os.path.join(self.pcd_path, fnn + '.npy'))
        norms = np.load(os.path.join(self.normals_path, fnn + '.npy'))

        lid = pcd[:,6]
        pcd = pcd[:,:6]

        p, c_p = pcd.shape

        pcd = pcd.transpose(1, 0)
        norms = norms.transpose(1, 0)
        norms = torch.from_numpy(norms.copy()).float()
        pcd = torch.from_numpy(pcd.copy()).float()
        lid = torch.from_numpy(lid.copy()).float()

        return pcd, norms, lid, fnn

    def __len__(self):
        return len(self.file_list)


if __name__ == '__main__':
    dataset = PcdIID(train=True)
    dataload = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=6)
    for ep in range(10):
        time1 = time.time()
        for i, data in enumerate(dataload):
            img, normal, fn = data
            print(img.shape)
            print(normal.shape)
