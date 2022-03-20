import cv2, time, imageio, json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from params import *

# NeRF MLP
class NeRF(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        input_dims = 2 * 3 * POS_ENCODE_DIMS + 3
        views_dims = 2 * 3 * VIEWS_ENCODE_DIMS + 3
        self.pts_linear = [nn.Linear(input_dims, CHSZ)]
        for i in range(num_layers - 1):
            if i == 4:
                self.pts_linear.append(nn.Linear(CHSZ + input_dims, CHSZ))
            else:
                self.pts_linear.append(nn.Linear(CHSZ, CHSZ))
        self.pts_linear = nn.ModuleList(self.pts_linear)

        self.views_linears = nn.Linear(views_dims + CHSZ, CHSZ//2)
        self.feature_linear = nn.Linear(CHSZ, CHSZ)
        self.alpha_linear = nn.Linear(CHSZ, 1)
        self.rgb_linear = nn.Linear(CHSZ//2, 3)

    def forward(self, x, v):
        h = torch.clone(x)
        for i, l in enumerate(self.pts_linear):
            h = l(h)
            h = F.relu(h)
            if i == 4:
                h = torch.cat([x, h], dim = -1)

        alpha = self.alpha_linear(h)
        h = self.feature_linear(h)
        h = torch.cat([h, v], dim = -1)
        h = self.views_linears(h)
        h = F.relu(h)
        rgb = self.rgb_linear(h)
        outputs = torch.cat([rgb, alpha], dim = -1)
        return outputs
