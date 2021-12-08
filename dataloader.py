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
    def __init__(self, train = True):
        self.train = train
        if self.train:
            data = np.load('./dataset/tiny_nerf_data.npz')
            self.focal = float(data['focal'])
            self.images = data['images']
            self.poses = data['poses']
        else:
            self.root = './dataset/lego/val/r_'
            self.load_images(self.root)
            file = json.load(open('./dataset/lego/transforms_val.json', 'r'))
            self.focal = (0.5 * DIMENSIONS) / np.tan(0.5 * file['camera_angle_x'])
            self.poses = file['frames']

    def load_images(self, root):
        self.images = []
        for i in range(100):
            self.images.append(cv2.imread(root + str(i) + '.png'))
        self.images = np.array(self.images)

    def __len__(self):
        return 100 * DIMENSIONS * DIMENSIONS

    def __getitem__(self, index):
        i, j = ((index // DIMENSIONS) % DIMENSIONS, index % DIMENSIONS)
        if self.train:
            pose = torch.tensor(self.poses[index // (DIMENSIONS * DIMENSIONS)], dtype = torch.float)
            img = torch.tensor(self.images[index // (DIMENSIONS * DIMENSIONS), min(i, 99), min(j, 99), :],
                               dtype = torch.float)
        else:
            pose = torch.tensor(self.poses[index // (DIMENSIONS * DIMENSIONS)]['transform_matrix'], dtype = torch.float)
            img = torch.tensor(self.images[index // (DIMENSIONS * DIMENSIONS), min(i, 799), min(j, 799), :],
                               dtype = torch.float) / 255
        ray = map_fn(i, j, self.focal, pose)
        return {'I': img, 'R': ray}

# Load Video Data
class VideoDataset(Dataset):
    def __init__(self):
        data = np.load('./dataset/tiny_nerf_data.npz')
        self.focal = float(data['focal'])

    def __len__(self):
        return 120 * DIMENSIONS * DIMENSIONS

    def get_translation_t(self, t):
        return torch.tensor([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, t],
                    [0, 0, 0, 1]], dtype = torch.float)

    def get_rotation_phi(self, phi):
        return torch.tensor([
                    [1, 0, 0, 0],
                    [0, np.cos(phi), -np.sin(phi), 0],
                    [0, np.sin(phi), np.cos(phi), 0],
                    [0, 0, 0, 1]], dtype = torch.float)

    def get_rotation_theta(self, theta):
        return torch.tensor([
                    [np.cos(theta), 0, -np.sin(theta), 0],
                    [0, 1, 0, 0],
                    [np.sin(theta), 0, np.cos(theta), 0],
                    [0, 0, 0, 1]], dtype = torch.float)

    def pose_spherical(self, theta, phi, t):
        c2w = self.get_translation_t(t)
        c2w = self.get_rotation_phi(phi / 180.0 * np.pi) @ c2w
        c2w = self.get_rotation_theta(theta / 180.0 * np.pi) @ c2w
        c2w = torch.tensor([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype = torch.float) @ c2w
        return c2w

    def __getitem__(self, index):
        i, j = ((index // DIMENSIONS) % DIMENSIONS, index % DIMENSIONS)
        pose = self.pose_spherical(3 * (index // (DIMENSIONS * DIMENSIONS)), -30.0, 4.0)
        img = torch.tensor([0.0, 0.0, 0.0])
        ray = map_fn(i, j, self.focal, pose)
        return {'I': img, 'R': ray}
