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

# Make experiment directory
if not os.path.exists('checkpoints/' + EXP_NAME):
    os.makedirs('checkpoints/' + EXP_NAME)

# Run the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = CustomDataset(train = True)
val_dataset = CustomDataset(train = False)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 8)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 8)

model = [NeRF(num_layers = NUM_LAYERS).to(device), NeRF(num_layers = NUM_LAYERS).to(device)]
if START != 0:
    if LATEST:
        checkpoint = torch.load('checkpoints/' + EXP_NAME + '/latest.pth')
    else:
        checkpoint = torch.load('checkpoints/' + EXP_NAME + '/' + str(START) + '.pth')
    model[0].load_state_dict(checkpoint['model_state_dict'][0])
    model[1].load_state_dict(checkpoint['model_state_dict'][1])
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(list(model[0].parameters()) + list(model[1].parameters()), lr = LR)
out_file = open('checkpoints/' + EXP_NAME +  '/losses.txt', 'a')

for epoch in range(START, EPOCHS):
    print('Starting Epoch ' + str(epoch + 1))

    for k, loader in enumerate([train_loader, val_loader]):
        if k == 1 and epoch % 100 != 99:
            continue
        
        current_loss_coarse = 0.0
        current_loss_fine = 0.0
        start = time.time()
    
        for i, data in enumerate(loader):
            image = data['I'].to(device)
            ray = data['R']

            if k == 0:
                optimizer.zero_grad()
            ray_flat, t_vals, ray_views, ray_origin, ray_direction = ray
            ray_flat, t_vals, ray_views, ray_origin, ray_direction = (ray_flat.to(device), t_vals.to(device),
                    ray_views.to(device), ray_origin.to(device), ray_direction.to(device))
            rgb, weights, predictions = render_rgb_depth(model[0], ray_flat, t_vals, ray_views, NUM_SAMPLES)
            loss_coarse = loss_function(image, rgb)
            t_vals, _ = torch.sort(torch.cat([sample_pdf((t_vals[..., 1:] + t_vals[..., :-1]) / 2, weights[..., 1:-1]).detach(),
                                              t_vals], dim = -1), dim = -1)
            (ray_flat, ray_views) = render_flat_rays(ray_origin, ray_direction, t_vals)
                    
            rgb, weights, _ = render_rgb_depth(model[1], ray_flat, t_vals, ray_views, NUM_FINE + NUM_SAMPLES)
            loss_fine = loss_function(image, rgb)
            loss = loss_coarse + loss_fine
            if k == 0:
                loss.backward()
                optimizer.step()
            current_loss_coarse += loss_coarse.item()
            current_loss_fine += loss_fine.item()

        if k == 0:
            for i, name in enumerate([str(epoch + 1), 'latest']):
                if i == 0 and epoch % 100 != 99:
                    continue
                elif i == 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = LR * (0.1 ** (epoch / (DECAY * 100)))
                torch.save({
                    'epoch': str(epoch + 1),
                    'model_state_dict': [model[0].state_dict(), model[1].state_dict()],
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, 'checkpoints/' + EXP_NAME + '/' + name + '.pth')

        for func in [out_file.write, print]:
            func('Loss after epoch ' + str(epoch + 1) + ': ' + str(current_loss_coarse * BATCH_SIZE / len(train_dataset)) + '\t'
                 + str(current_loss_fine * BATCH_SIZE / len(train_dataset)) + '\tTime Taken: ' + str(time.time() - start))
