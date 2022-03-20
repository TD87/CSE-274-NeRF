import cv2, time, imageio, json, time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from params import *

# Fourier encoding of input
def encode_position(x, views = False):
    positions = [x]
    dims = POS_ENCODE_DIMS if views == False else VIEWS_ENCODE_DIMS
    for i in range(dims):
        positions.append(torch.sin((2.0 ** i) * x))
        positions.append(torch.cos((2.0 ** i) * x))
    return torch.cat(positions, dim = -1)

# Generate ray origins and directions for a given camera
def get_rays(i, j, focal, pose):
    ray_origin = pose[:3, -1]
    camera_matrix = pose[:3, :3]
    transformed_i = (i - (0.5 * DIMENSIONS)) / focal
    transformed_j = (j - (0.5 * DIMENSIONS)) / focal
    direction = torch.tensor([transformed_i, transformed_j, 1], dtype = torch.float)
    ray_direction = camera_matrix @ direction
    ray_direction = ray_direction / torch.norm(ray_direction)
    return (ray_origin, ray_direction)

# Generate samples along the ray
def render_flat_rays(ray_origin, ray_direction, t_vals):
    ray = ray_origin.unsqueeze(-2) + (ray_direction.unsqueeze(-2) * t_vals.unsqueeze(-1))
    ray_views = torch.ones_like(ray) * ray_direction.unsqueeze(-2)
    ray_views = encode_position(ray_views, views = True)
    ray_flat = encode_position(ray)
    return (ray_flat, ray_views)

# Map camera params to rays samples
def map_fn(i, j, focal, pose):
    (ray_origin, ray_direction) = get_rays(i, j, focal, pose)
    t_vals = torch.linspace(NEAR, FAR, NUM_SAMPLES)
    noise = torch.rand(NUM_SAMPLES) * (FAR - NEAR) / NUM_SAMPLES
    t_vals = t_vals + noise
    (ray_flat, ray_views) = render_flat_rays(ray_origin.unsqueeze(0), ray_direction.unsqueeze(0), t_vals.unsqueeze(0))
    return (ray_flat[0], t_vals, ray_views[0], ray_origin, ray_direction)

# Volume Rendering
def render_rgb_depth(model, ray_flat, t_vals, ray_views, num_samples):
    predictions = model(ray_flat, ray_views)
    rgb = torch.sigmoid(predictions[:, :, :-1])
    sigma_a = F.relu(predictions[:, :, -1])

    delta = t_vals[:, 1:] - t_vals[:, :-1]
    delta = torch.cat([delta, 1e10 * torch.ones((BATCH_SIZE, 1)).to(delta.device)], dim = -1)
    alpha = 1.0 - torch.exp(-sigma_a * delta)

    exp_term = (1.0 - alpha) + 1e-10
    exp_term = torch.cat([torch.ones((BATCH_SIZE, 1)).to(alpha.device), exp_term[:, :-1]], dim = -1)
    transmittance = torch.cumprod(exp_term, dim = -1)
    weights = alpha * transmittance
    rgb = torch.sum(weights.unsqueeze(-1) * rgb, dim = -2)
    acc = torch.sum(weights, dim = -1)
    rgb = rgb + (1 - acc.unsqueeze(-1))
    # depth = torch.sum(weights * t_vals, dim = -1)
    return rgb, weights, predictions

# Sample fine samples from coarse samples weights
def sample_pdf(bins, weights):
    # According to (https://github.com/yenchenlin/nerf-pytorch/blob/master/run_nerf_helpers.py)
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)
    u = torch.rand(list(cdf.shape[:-1]) + [NUM_FINE]).to(cdf.device)

    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])
    return samples
