import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_spatial_position_encoding(H, W, d_model):
    pe = torch.zeros(H, W, d_model)
    y_pos = torch.arange(H).float().unsqueeze(1).repeat(1, W)
    x_pos = torch.arange(W).float().unsqueeze(0).repeat(H, 1)
    
    if H > 1:
        y_pos = 2 * (y_pos / (H - 1)) - 1
    else:
        y_pos = torch.zeros_like(y_pos)
    if W > 1:
        x_pos = 2 * (x_pos / (W - 1)) - 1
    else:
        x_pos = torch.zeros_like(x_pos)
    
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                         -(math.log(10000.0) / d_model))
    
    pe[:, :, 0::2] = torch.sin(y_pos.unsqueeze(-1) * div_term)
    pe[:, :, 1::2] = torch.cos(x_pos.unsqueeze(-1) * div_term)
    
    return pe

def compute_distance_map(H, W, sigma=1.0):
    y_coords = torch.arange(H).float().unsqueeze(1).repeat(1, W)
    x_coords = torch.arange(W).float().unsqueeze(0).repeat(H, 1)
    coords = torch.stack([y_coords.flatten(), x_coords.flatten()], dim=1)
    distances = torch.cdist(coords, coords)
    distance_weights = torch.exp(-distances / sigma)
    return distance_weights

def make_norm(norm_type: str, channels: int):
    if norm_type == "gn":
        for g in [32, 16, 8, 4, 2]:
            if channels % g == 0:
                return nn.GroupNorm(g, channels)
        return nn.GroupNorm(1, channels)
    elif norm_type == "bn":
        return nn.BatchNorm2d(channels)
    else:
        return nn.Identity()

def _make_gn(channels: int):
    for g in [32, 16, 8, 4, 2]:
        if channels % g == 0:
            return nn.GroupNorm(g, channels)
    return nn.GroupNorm(1, channels)

def window_partition(x, window_size):
    B, C, H, W = x.shape
    assert H % window_size == 0 and W % window_size == 0
    x = x.view(B, C,
               H // window_size, window_size,
               W // window_size, window_size)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    x = x.view(-1, C, window_size, window_size)
    return x


def window_reverse(x, window_size, H, W):
    Bn, C, wH, wW = x.shape
    nH = H // window_size
    nW = W // window_size
    B = Bn // (nH * nW)
    x = x.view(B, nH, nW, C, wH, wW)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    x = x.view(B, C, H, W)
    return x


def pad_to_multiple(x, multiple: int):
    B, C, H, W = x.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    x = F.pad(x, (0, pad_w, 0, pad_h))
    return x, (pad_h, pad_w)

def unpad(x, pad_h: int, pad_w: int):
    if pad_h > 0:
        x = x[:, :, :-pad_h, :]
    if pad_w > 0:
        x = x[:, :, :, :-pad_w]
    return x

def build_shift_mask(Hp, Wp, window_size, shift_size, device):
    img_mask = torch.zeros((1, Hp, Wp, 1), device=device)
    h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
    
    mask = img_mask.permute(0, 3, 1, 2).contiguous()
    mask_windows = window_partition(mask, window_size)
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
    return attn_mask

