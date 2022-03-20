import cv2, time, imageio, json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from params import *
from utils import *       

# Load Data
class CustomDataset(Dataset):
    def __init__(self, train = None):
        self.train = train
        self.iroot = './bottles/rgb/'
        self.proot = './bottles/pose/'
        self.focal = 875.0 / (800 / DIMENSIONS)
        if self.train is None:
            self.poses = [self.load_txt(self.proot + '2_test_0000.txt'), self.load_txt(self.proot + '2_test_0016.txt'),
                          self.load_txt(self.proot + '2_test_0055.txt'), self.load_txt(self.proot + '2_test_0093.txt'),
                          self.load_txt(self.proot + '2_test_0160.txt')]
        elif self.train:
            self.load_data(self.iroot + '0_train_', self.proot + '0_train_')
        elif not self.train:   
            self.load_data(self.iroot + '1_val_', self.proot + '1_val_')
            
    def load_txt(self, path):
        pose = []
        with open(path, 'r') as file:
            for line in file.readlines():
                pose.append([])
                for num in line.split():
                    pose[-1].append(float(num))
        pose = np.array(pose)
        return pose  

    def load_data(self, iroot, proot):
        self.images, self.poses = [], []
        for i in range(100):
            im = cv2.imread(iroot + str(i).zfill(4) + '.png')
            if DIMENSIONS < 800:
                im = cv2.resize(im, (DIMENSIONS, DIMENSIONS), cv2.INTER_AREA)
            self.images.append(im)
            self.poses.append(self.load_txt(proot + str(i).zfill(4) + '.txt'))
        self.images = np.array(self.images)
        self.poses = np.array(self.poses)

    def __len__(self):
        if self.train is None:
            return 5 * DIMENSIONS * DIMENSIONS
        else:
            return 10000

    def __getitem__(self, index):
        if self.train is not None:
            index = np.random.randint(100 * DIMENSIONS * DIMENSIONS)
        i, j = ((index // DIMENSIONS) % DIMENSIONS, index % DIMENSIONS)
        pose = torch.tensor(self.poses[index // (DIMENSIONS * DIMENSIONS)], dtype = torch.float)
        if self.train is not None:
            img = torch.tensor(self.images[index // (DIMENSIONS * DIMENSIONS), min(j, DIMENSIONS - 1), 
                                           min(i, DIMENSIONS - 1), :], dtype = torch.float) / 255
        else:
            img = 0
        ray = map_fn(i, j, self.focal, pose)
        return {'I': img, 'R': ray}
