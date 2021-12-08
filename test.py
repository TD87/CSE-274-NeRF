import cv2, time, imageio, json
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

# Run the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if VIDEO:
    validation_dataset = VideoDataset()
else:
    validation_dataset = CustomDataset(train = False)
validation_loader = DataLoader(validation_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 8)

model = NeRF(num_layers = NUM_LAYERS).to(device)
checkpoint = torch.load('checkpoints/joint/latest.pth')
model.load_state_dict(checkpoint['model_state_dict'])
final = torch.zeros((DIMENSIONS, DIMENSIONS, 3), dtype = torch.float)
frames = []

for i, data in enumerate(validation_loader):
    image = data['I'].to(device)
    ray = data['R']

    for j in range(2):
        if j == 0:
            ray_flat, t_vals, ray_views, ray_origin, ray_direction = ray
            ray_flat, t_vals, ray_views, ray_origin, ray_direction = (ray_flat.to(device), t_vals.to(device),
                    ray_views.to(device), ray_origin.to(device), ray_direction.to(device))
            rgb, weights = render_rgb_depth(model, ray_flat, t_vals, ray_views, NUM_SAMPLES)
            t_vals, _ = torch.sort(torch.cat([sample_pdf(t_vals, weights), t_vals], dim = -1), dim = -1)
            (ray_flat, ray_views) = render_flat_rays(ray_origin, ray_direction, t_vals)
        else:
            rgb, weights = render_rgb_depth(model, ray_flat, t_vals, ray_views, NUM_FINE + NUM_SAMPLES)

    for j in range(BATCH_SIZE):
        row, col = (((i * BATCH_SIZE + j) // DIMENSIONS) % DIMENSIONS, (i * BATCH_SIZE + j) % DIMENSIONS)
        final[row, col, :] = rgb.cpu().detach()[j]
        if (i * BATCH_SIZE + j) % (DIMENSIONS * DIMENSIONS) == (DIMENSIONS * DIMENSIONS) - 1:
            plt.imshow(final.numpy())
            plt.savefig('./results/joint/validation/' + str(((i * BATCH_SIZE + j) // (DIMENSIONS * DIMENSIONS)) + 1) + '.jpg')
            if VIDEO:
                frames.append(np.clip(255 * final.numpy(), 0.0, 255.0).astype(np.uint8))

if VIDEO:
    rgb_video = "./results/rgb_video.mp4"
    imageio.mimwrite(rgb_video, frames, fps=10, quality=7, macro_block_size=None)
