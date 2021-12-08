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
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = CustomDataset(train = True)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 8)

model = [NeRF(num_layers = NUM_LAYERS).to(device), NeRF(num_layers = NUM_LAYERS).to(device)]
if START != 0:
    if LATEST:
        checkpoint = torch.load('checkpoints/dual/latest.pth')
    else:
        checkpoint = torch.load('checkpoints/dual/' + str(START) + '.pth')
    model[0].load_state_dict(checkpoint['model_state_dict'][0])
    model[1].load_state_dict(checkpoint['model_state_dict'][1])
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(list(model[0].parameters()) + list(model[1].parameters()), lr = LR)
out_file = open('losses/losses.txt', 'a')

for epoch in range(START, EPOCHS):
    print('Starting Epoch ' + str(epoch + 1))
    current_loss = 0.0
    start = time.time()

    for i, data in enumerate(train_loader):
        image = data['I'].to(device)
        ray = data['R']

        optimizer.zero_grad()
        for j in range(2):
            if j == 0:
                ray_flat, t_vals, ray_views, ray_origin, ray_direction = ray
                ray_flat, t_vals, ray_views, ray_origin, ray_direction = (ray_flat.to(device), t_vals.to(device),
                        ray_views.to(device), ray_origin.to(device), ray_direction.to(device))
                rgb, weights = render_rgb_depth(model[0], ray_flat, t_vals, ray_views, NUM_SAMPLES)
                loss = loss_function(image, rgb)
                t_vals, _ = torch.sort(torch.cat([sample_pdf(t_vals, weights.detach()), t_vals], dim = -1), dim = -1)
                (ray_flat, ray_views) = render_flat_rays(ray_origin, ray_direction, t_vals)
            else:
                rgb, weights = render_rgb_depth(model[1], ray_flat, t_vals, ray_views, NUM_FINE + NUM_SAMPLES)
                loss += loss_function(image, rgb)
                loss.backward()
                optimizer.step()
                current_loss += loss.item()

    if epoch % 50 == 49:
        torch.save({
            'epoch': str(epoch + 1),
            'model_state_dict': [model[0].state_dict(), model[1].state_dict()],
            'optimizer_state_dict': optimizer.state_dict(),
            }, 'checkpoints/dual/' + str(epoch + 1) + '.pth')
    torch.save({
        'epoch': str(epoch + 1),
        'model_state_dict': [model[0].state_dict(), model[1].state_dict()],
        'optimizer_state_dict': optimizer.state_dict(),
        }, 'checkpoints/dual/latest.pth')

    out_file.write('Loss after epoch ' + str(epoch + 1) + ': ' + str(current_loss * BATCH_SIZE / len(train_dataset))
          + '\tTime Taken: ' + str(time.time() - start) + '\n')
    print('Loss after epoch ' + str(epoch + 1) + ': ' + str(current_loss * BATCH_SIZE / len(train_dataset))
          + '\tTime Taken: ' + str(time.time() - start))
