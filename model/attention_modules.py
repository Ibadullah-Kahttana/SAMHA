import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init

from model.utils import (
    get_spatial_position_encoding,
    compute_distance_map,
    window_partition,
    window_reverse,
)

from model.fusion_modules import (
    LFIFusion,
    SEFusion,
    CBAMFusion,
    AdaptiveFusion,
    CrossAttentionFusion,
    SimpleWeightedFusion,
)

class SAMHAChannelGate(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.fc_dual = nn.Sequential(
            nn.Linear(channels * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )
        self.fc_triple = nn.Sequential(
            nn.Linear(channels * 3, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, local, medium, large=None):
        B, C, H, W = local.shape

        l = local.mean(dim=[2, 3])
        m = medium.mean(dim=[2, 3])

        if large is not None:
            g = large.mean(dim=[2, 3])
            ctx = torch.cat([l, m, g], dim=1)
            w = self.fc_triple(ctx)
            w = w.view(B, C, 1, 1)
            fused = local * w + medium * (1 - w) * 0.5 + large * (1 - w) * 0.5
        else:
            ctx = torch.cat([l, m], dim=1)
            w = self.fc_dual(ctx)
            w = w.view(B, C, 1, 1)
            fused = local * w + medium * (1 - w)

        return fused

class SAMHAWindow(nn.Module):
    def __init__(self, channels, num_heads=4, window_size=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0

        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)
        self.out_proj = nn.Conv2d(channels, channels, 1)

    def _pad_to_window_size(self, x):
        B, C, H, W = x.shape
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x, H, W, pad_h, pad_w

    def forward(self, local, medium, large=None):
        B, C, H, W = local.shape
        needs_pad = (H % self.window_size != 0) or (W % self.window_size != 0)

        if needs_pad:
            local_padded, _, _, pad_h, pad_w = self._pad_to_window_size(local)
            medium_padded, _, _, _, _ = self._pad_to_window_size(medium)
            if large is not None:
                large_padded, _, _, _, _ = self._pad_to_window_size(large)
                ctx = 0.5 * (medium_padded + large_padded)
            else:
                ctx = medium_padded
            _, _, H_pad, W_pad = local_padded.shape
        else:
            local_padded = local
            if large is not None:
                ctx = 0.5 * (medium + large)
            else:
                ctx = medium
            H_pad, W_pad = H, W

        q = self.q_proj(local_padded)
        k = self.k_proj(ctx)
        v = self.v_proj(ctx)

        q = window_partition(q, self.window_size)
        k = window_partition(k, self.window_size)
        v = window_partition(v, self.window_size)

        Bn, C, wH, wW = q.shape
        N = wH * wW

        q = q.view(Bn, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        k = k.view(Bn, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        v = v.view(Bn, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = out.permute(0, 1, 3, 2).contiguous().view(Bn, C, wH, wW)

        out = window_reverse(out, self.window_size, H_pad, W_pad)
        out = self.out_proj(out)

        if needs_pad:
            out = out[:, :, :H, :W]

        return local + out

class FuseLocalAndContext(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(2 * channels, channels, kernel_size=1)
        self.bn   = nn.BatchNorm2d(channels)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, local, fused):
        x = torch.cat([local, fused], dim=1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class ModularSAMHA(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_heads=8,
        d_model=512,
        fusion_type='lfi',
        lr_mult=None,
        weight_init_scale=1.0,
        distance_prior='exp',
        distance_sigma=1.0,
        lambda_dist_init=0.1,
        lambda_dist_trainable=True,
    ):
        super(ModularSAMHA, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_init_scale = weight_init_scale
        self.fusion_type = fusion_type
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.conv_query = nn.Conv2d(in_channels, d_model, 1)
        self.conv_key = nn.Conv2d(in_channels, d_model, 1)
        self.conv_value = nn.Conv2d(in_channels, d_model, 1)
        self.conv_out = nn.Conv2d(d_model, out_channels, 1)
        
        self.distance_prior = distance_prior
        self.distance_sigma = distance_sigma
        if lambda_dist_trainable:
            self.lambda_dist = nn.Parameter(torch.tensor(lambda_dist_init))
        else:
            self.register_buffer("lambda_dist", torch.tensor(lambda_dist_init))
        self.gamma = nn.Parameter(torch.tensor(1e-3))
        self.gamma_1 = nn.Parameter(torch.tensor(1e-3))
        self.gamma_2 = nn.Parameter(torch.tensor(1e-3))

        self.fusion_module = self._create_fusion_module(fusion_type, d_model)
        
        self.register_buffer('pos_encoding_cache', None)
        self.register_buffer('distance_map_cache', None)
        self.cached_H = None
        self.cached_W = None
        
        self.reset_parameters()
        self.reset_lr_mult(lr_mult)
        self.reset_weight_and_weight_decay()
    
    def _create_fusion_module(self, fusion_type, d_model):
        fusion_modules = {
            'lfi': LFIFusion,
            'se_fusion': SEFusion,
            'cbam_fusion': CBAMFusion,
            'adaptive_fusion': AdaptiveFusion,
            'cross_attention_fusion': CrossAttentionFusion,
            'simple_weighted': SimpleWeightedFusion,
        }
        
        if fusion_type not in fusion_modules:
            raise ValueError(
                f"Unknown fusion type: {fusion_type}. "
                f"Available: {list(fusion_modules.keys())}"
            )
        
        module_class = fusion_modules[fusion_type]
        
        if fusion_type == 'se_fusion':
            return module_class(d_model, reduction=16)
        elif fusion_type == 'cbam_fusion':
            return module_class(d_model, reduction=16)
        elif fusion_type == 'cross_attention_fusion':
            return module_class(d_model, num_heads=8)
        else:
            return module_class(d_model)
    
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    init.zeros_(m.bias)
    
    def reset_lr_mult(self, lr_mult):
        if lr_mult is not None:
            for m in self.modules():
                m.lr_mult = lr_mult
    
    def reset_weight_and_weight_decay(self):
        init.normal_(self.conv_query.weight, 0, 0.01 * self.weight_init_scale)
        init.normal_(self.conv_key.weight, 0, 0.01 * self.weight_init_scale)
        if hasattr(self.conv_query.weight, 'wd'):
            self.conv_query.weight.wd = 0.0
        if hasattr(self.conv_query.bias, 'wd'):
            self.conv_query.bias.wd = 0.0
        if hasattr(self.conv_key.weight, 'wd'):
            self.conv_key.weight.wd = 0.0

    def _distance_bias(self, distance_map):
        if self.distance_prior in (None, "none"):
            return None

        eps = 1e-6
        if self.distance_prior == "log":
            bias_base = torch.log(distance_map + eps)
        elif self.distance_prior == "exp":
            bias_base = distance_map
        elif self.distance_prior == "inv":
            dist = -self.distance_sigma * torch.log(distance_map + eps)
            bias_base = 1.0 / (dist + eps)
        elif self.distance_prior == "gaussian":
            dist = -self.distance_sigma * torch.log(distance_map + eps)
            bias_base = torch.exp(-(dist ** 2) / (2.0 * (self.distance_sigma ** 2)))
        elif self.distance_prior == "raw":
            dist = -self.distance_sigma * torch.log(distance_map + eps)
            bias_base = dist
        else:
            raise ValueError(f"Unknown distance_prior: {self.distance_prior}")

        return self.lambda_dist * bias_base

    def forward(self, x, y=None, z=None):
        B, C, H, W = x.size()
        residual = x
        
        if (self.pos_encoding_cache is None or 
            self.cached_H != H or self.cached_W != W):
            pos_encoding = get_spatial_position_encoding(H, W, self.d_model)
            distance_map = compute_distance_map(H, W, sigma=self.distance_sigma)
            pos_encoding = pos_encoding.to(device=x.device, dtype=x.dtype)
            distance_map = distance_map.to(device=x.device, dtype=x.dtype)
            self.pos_encoding_cache = pos_encoding
            self.distance_map_cache = distance_map
            self.cached_H = H
            self.cached_W = W
        else:
            pos_encoding = self.pos_encoding_cache
            distance_map = self.distance_map_cache
        
        pos_encoding = pos_encoding.unsqueeze(0).expand(B, -1, -1, -1)
        pos_encoding = pos_encoding.permute(0, 3, 1, 2)
        
        Q = self.conv_query(x) + pos_encoding
        Q = Q.view(B, self.num_heads, self.d_k, H * W).transpose(2, 3)
    
        outputs = []
        
        if y is not None:
            K_y = self.conv_key(y) + pos_encoding
            V_y = self.conv_value(y)
            K_y = K_y.view(B, self.num_heads, self.d_k, H * W).transpose(2, 3)
            V_y = V_y.view(B, self.num_heads, self.d_k, H * W).transpose(2, 3)
            
            scores_y = torch.matmul(Q, K_y.transpose(-2, -1)) / math.sqrt(self.d_k)
            distance_bias = self._distance_bias(distance_map)
            if distance_bias is not None:
                scores_y = scores_y + distance_bias.unsqueeze(0).unsqueeze(0)
            attn_y = F.softmax(scores_y, dim=-1)
            
            out_y = torch.matmul(attn_y, V_y)
            out_y = out_y.transpose(2, 3).contiguous().view(B, self.d_model, H, W)
            outputs.append(out_y)
        
        if z is not None:
            K_z = self.conv_key(z) + pos_encoding
            V_z = self.conv_value(z)
            K_z = K_z.view(B, self.num_heads, self.d_k, H * W).transpose(2, 3)
            V_z = V_z.view(B, self.num_heads, self.d_k, H * W).transpose(2, 3)
            
            scores_z = torch.matmul(Q, K_z.transpose(-2, -1)) / math.sqrt(self.d_k)
            distance_bias = self._distance_bias(distance_map)
            if distance_bias is not None:
                scores_z = scores_z + distance_bias.unsqueeze(0).unsqueeze(0)
            attn_z = F.softmax(scores_z, dim=-1)
            
            out_z = torch.matmul(attn_z, V_z)
            out_z = out_z.transpose(2, 3).contiguous().view(B, self.d_model, H, W)
            outputs.append(out_z)
        
        K_x = self.conv_key(x) + pos_encoding
        V_x = self.conv_value(x)
        K_x = K_x.view(B, self.num_heads, self.d_k, H * W).transpose(2, 3)
        V_x = V_x.view(B, self.num_heads, self.d_k, H * W).transpose(2, 3)
        
        scores_x = torch.matmul(Q, K_x.transpose(-2, -1)) / math.sqrt(self.d_k)
        distance_bias = self._distance_bias(distance_map)
        if distance_bias is not None:
            scores_x = scores_x + distance_bias.unsqueeze(0).unsqueeze(0)
        attn_x = F.softmax(scores_x, dim=-1)
        
        out_x = torch.matmul(attn_x, V_x)
        out_x = out_x.transpose(2, 3).contiguous().view(B, self.d_model, H, W)
        outputs.append(out_x)
        
        fused = self.fusion_module(outputs, self.gamma, self.gamma_1, self.gamma_2)
        out = self.conv_out(fused)
        return residual + out

class SAMHA(ModularSAMHA):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_heads=8,
        d_model=512,
        fusion_type='lfi',
        lr_mult=None,
        weight_init_scale=1.0,
        distance_prior='log',
        distance_sigma=1.0,
        lambda_dist_init=0.1,
        lambda_dist_trainable=True,
    ):
        super(SAMHA, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_heads=num_heads,
            d_model=d_model,
            fusion_type=fusion_type,
            lr_mult=lr_mult,
            weight_init_scale=weight_init_scale,
            distance_prior=distance_prior,
            distance_sigma=distance_sigma,
            lambda_dist_init=lambda_dist_init,
            lambda_dist_trainable=lambda_dist_trainable,
        )
