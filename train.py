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

model = NeRF(num_layers = NUM_LAYERS).to(device)
if START != 0:
    if LATEST:
        checkpoint = torch.load('checkpoints/joint/latest.pth')
    else:
        checkpoint = torch.load('checkpoints/joint/' + str(START) + '.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = LR)
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
                rgb, weights = render_rgb_depth(model, ray_flat, t_vals, ray_views, NUM_SAMPLES)
                t_vals, _ = torch.sort(torch.cat([sample_pdf(t_vals, weights.detach()), t_vals], dim = -1), dim = -1)
                (ray_flat, ray_views) = render_flat_rays(ray_origin, ray_direction, t_vals)
            else:
                rgb, _ = render_rgb_depth(model, ray_flat, t_vals, ray_views, NUM_FINE + NUM_SAMPLES)

            loss = loss_function(image, rgb)
            loss.backward()
            optimizer.step()
            current_loss += loss.item()

    if epoch % 50 == 49:
        torch.save({
            'epoch': str(epoch + 1),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, 'checkpoints/joint/' + str(epoch + 1) + '.pth')
    torch.save({
        'epoch': str(epoch + 1),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, 'checkpoints/joint/latest.pth')

    out_file.write('Loss after epoch ' + str(epoch + 1) + ': ' + str(current_loss * BATCH_SIZE / len(train_dataset))
          + '\tTime Taken: ' + str(time.time() - start) + '\n')
    print('Loss after epoch ' + str(epoch + 1) + ': ' + str(current_loss * BATCH_SIZE / len(train_dataset))
          + '\tTime Taken: ' + str(time.time() - start))
