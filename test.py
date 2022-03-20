import cv2, time, imageio, json, os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from params import *
from utils import *
from dataloader import *
from model import *

# Make results directory
if not os.path.exists('results/' + EXP_NAME):
    os.makedirs('results/' + EXP_NAME)

# Run the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_dataset = CustomDataset()
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 8)

model = [NeRF(num_layers = NUM_LAYERS).to(device), NeRF(num_layers = NUM_LAYERS).to(device)]
checkpoint = torch.load('checkpoints/' + EXP_NAME + '/latest.pth')
model[0].load_state_dict(checkpoint['model_state_dict'][0])
model[1].load_state_dict(checkpoint['model_state_dict'][1])
final = torch.zeros((DIMENSIONS, DIMENSIONS, 3), dtype = torch.float)
frames = []

for i, data in enumerate(test_loader):
    ray = data['R']

    ray_flat, t_vals, ray_views, ray_origin, ray_direction = ray
    ray_flat, t_vals, ray_views, ray_origin, ray_direction = (ray_flat.to(device), t_vals.to(device),
            ray_views.to(device), ray_origin.to(device), ray_direction.to(device))
    rgb, weights, predictions = render_rgb_depth(model[0], ray_flat, t_vals, ray_views, NUM_SAMPLES)
    t_vals, _ = torch.sort(torch.cat([sample_pdf((t_vals[..., 1:] + t_vals[..., :-1]) / 2, weights[..., 1:-1]).detach(),
                                      t_vals], dim = -1), dim = -1)
    (ray_flat, ray_views) = render_flat_rays(ray_origin, ray_direction, t_vals)                
    rgb, weights, _ = render_rgb_depth(model[1], ray_flat, t_vals, ray_views, NUM_FINE + NUM_SAMPLES)

    for j in range(BATCH_SIZE):
        col, row = (((i * BATCH_SIZE + j) // DIMENSIONS) % DIMENSIONS, (i * BATCH_SIZE + j) % DIMENSIONS)
        final[row, col, :] = rgb.cpu().detach()[j]
        if (i * BATCH_SIZE + j) % (DIMENSIONS * DIMENSIONS) == (DIMENSIONS * DIMENSIONS) - 1:
            save = final.numpy()
            save[save > 1] = 1
            save[save < 0] = 0
            save = np.uint8(save * 255)
            cv2.imwrite('./results/' + EXP_NAME + '/' + str(((i * BATCH_SIZE + j) // (DIMENSIONS * DIMENSIONS)) + 1) + '.png', save)
