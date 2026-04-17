import torch
import torch.nn as nn
from torch.nn import init
import math
from model.blocks import SEBlock, CBAMBlock

class BaseFusionModule(nn.Module):
    def forward(self, outputs, gamma, gamma_1, gamma_2):
        raise NotImplementedError

class LFIFusion(BaseFusionModule):
    def __init__(self, d_model):
        super(LFIFusion, self).__init__()
        self.d_model = d_model
        
        self.in_1 = nn.Conv2d(d_model, d_model, 1)
        self.in_2 = nn.Conv2d(d_model, d_model, 1)
        self.in_3 = nn.Conv2d(d_model, d_model, 1)
        self.trans = nn.Conv2d(d_model * 3, d_model * 3, 1)
        self.out_1 = nn.Conv2d(d_model, d_model, 1)
        self.out_2 = nn.Conv2d(d_model, d_model, 1)
        self.out_3 = nn.Conv2d(d_model, d_model, 1)
        self.softmax_H = nn.Softmax(dim=0)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    init.zeros_(m.bias)
    
    def forward(self, outputs, gamma, gamma_1, gamma_2):
        if len(outputs) == 3:
            out_sim = gamma * outputs[0]
            out_sim_1 = gamma_1 * outputs[1]
            out_sim_2 = gamma_2 * outputs[2]
            
            H_1 = self.in_1(out_sim)
            H_2 = self.in_2(out_sim_1)
            H_3 = self.in_3(out_sim_2)
            H_cat = torch.cat((H_1, H_2, H_3), 1)
            H_tra = self.trans(H_cat)
            H_spl = torch.split(H_tra, self.d_model, dim=1)
            H_4 = torch.sigmoid(self.out_1(H_spl[0]))
            H_5 = torch.sigmoid(self.out_2(H_spl[1]))
            H_6 = torch.sigmoid(self.out_3(H_spl[2]))
            H_st = torch.stack((H_4, H_5, H_6), 0)
            H_all = self.softmax_H(H_st)
            
            fused = H_all[0] * out_sim + H_all[1] * out_sim_1 + H_all[2] * out_sim_2
        elif len(outputs) == 2:
            out_sim = gamma * outputs[0]
            out_sim_1 = gamma_1 * outputs[1]
            out_sim_2 = gamma_2 * outputs[1]
            
            H_1 = self.in_1(out_sim)
            H_2 = self.in_2(out_sim_1)
            H_3 = self.in_3(out_sim_2)
            H_cat = torch.cat((H_1, H_2, H_3), 1)
            H_tra = self.trans(H_cat)
            H_spl = torch.split(H_tra, self.d_model, dim=1)
            H_4 = torch.sigmoid(self.out_1(H_spl[0]))
            H_5 = torch.sigmoid(self.out_2(H_spl[1]))
            H_6 = torch.sigmoid(self.out_3(H_spl[2]))
            H_st = torch.stack((H_4, H_5, H_6), 0)
            H_all = self.softmax_H(H_st)
            
            fused = H_all[0] * out_sim + H_all[1] * out_sim_1 + H_all[2] * out_sim_2
        elif len(outputs) == 1:
            fused = gamma * outputs[0]
        else:
            fused = torch.zeros_like(outputs[0]) if outputs else None
        return fused

class SEFusion(BaseFusionModule):
    def __init__(self, d_model, reduction=16):
        super(SEFusion, self).__init__()
        self.d_model = d_model
        self.se_1 = SEBlock(d_model, reduction)
        self.se_2 = SEBlock(d_model, reduction)
        self.se_3 = SEBlock(d_model, reduction)
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, outputs, gamma, gamma_1, gamma_2):
        if len(outputs) == 3:
            out_sim = gamma * outputs[0]
            out_sim_1 = gamma_1 * outputs[1]
            out_sim_2 = gamma_2 * outputs[2]
            
            out_sim = self.se_1(out_sim)
            out_sim_1 = self.se_2(out_sim_1)
            out_sim_2 = self.se_3(out_sim_2)
            
            weights = self.softmax(self.fusion_weights)
            fused = weights[0] * out_sim + weights[1] * out_sim_1 + weights[2] * out_sim_2
        elif len(outputs) == 2:
            out_sim = gamma * outputs[0]
            out_sim_1 = gamma_1 * outputs[1]
            out_sim_2 = gamma_2 * outputs[1]
            
            out_sim = self.se_1(out_sim)
            out_sim_1 = self.se_2(out_sim_1)
            out_sim_2 = self.se_3(out_sim_2)
            
            weights = self.softmax(self.fusion_weights)
            fused = weights[0] * out_sim + weights[1] * out_sim_1 + weights[2] * out_sim_2
        elif len(outputs) == 1:
            fused = gamma * outputs[0]
        else:
            fused = torch.zeros_like(outputs[0]) if outputs else None
        return fused

class CBAMFusion(BaseFusionModule):
    def __init__(self, d_model, reduction=16):
        super(CBAMFusion, self).__init__()
        self.d_model = d_model
        self.cbam_1 = CBAMBlock(d_model, reduction)
        self.cbam_2 = CBAMBlock(d_model, reduction)
        self.cbam_3 = CBAMBlock(d_model, reduction)
        
        self.fusion_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(d_model * 3, d_model, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model, 3, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, outputs, gamma, gamma_1, gamma_2):
        if len(outputs) == 3:
            out_sim = gamma * outputs[0]
            out_sim_1 = gamma_1 * outputs[1]
            out_sim_2 = gamma_2 * outputs[2]
            
            out_sim = self.cbam_1(out_sim)
            out_sim_1 = self.cbam_2(out_sim_1)
            out_sim_2 = self.cbam_3(out_sim_2)
            
            concat_features = torch.cat([out_sim, out_sim_1, out_sim_2], dim=1)
            weights = self.fusion_net(concat_features)
            
            fused = (weights[:, 0:1] * out_sim + 
                    weights[:, 1:2] * out_sim_1 + 
                    weights[:, 2:3] * out_sim_2)
        elif len(outputs) == 2:
            out_sim = gamma * outputs[0]
            out_sim_1 = gamma_1 * outputs[1]
            out_sim_2 = gamma_2 * outputs[1]
            
            out_sim = self.cbam_1(out_sim)
            out_sim_1 = self.cbam_2(out_sim_1)
            out_sim_2 = self.cbam_3(out_sim_2)
            
            concat_features = torch.cat([out_sim, out_sim_1, out_sim_2], dim=1)
            weights = self.fusion_net(concat_features)
            
            fused = (weights[:, 0:1] * out_sim + 
                    weights[:, 1:2] * out_sim_1 + 
                    weights[:, 2:3] * out_sim_2)
        elif len(outputs) == 1:
            fused = gamma * outputs[0]
        else:
            fused = torch.zeros_like(outputs[0]) if outputs else None
        return fused

class AdaptiveFusion(BaseFusionModule):
    def __init__(self, d_model):
        super(AdaptiveFusion, self).__init__()
        self.d_model = d_model
        self.transform_1 = nn.Conv2d(d_model, d_model, 1)
        self.transform_2 = nn.Conv2d(d_model, d_model, 1)
        self.transform_3 = nn.Conv2d(d_model, d_model, 1)
        
        self.weight_net = nn.Sequential(
            nn.Conv2d(d_model * 3, d_model, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model, d_model // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model // 4, 3, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, outputs, gamma, gamma_1, gamma_2):
        if len(outputs) == 3:
            out_sim = gamma * outputs[0]
            out_sim_1 = gamma_1 * outputs[1]
            out_sim_2 = gamma_2 * outputs[2]
            
            out_sim = self.transform_1(out_sim)
            out_sim_1 = self.transform_2(out_sim_1)
            out_sim_2 = self.transform_3(out_sim_2)
            
            concat_features = torch.cat([out_sim, out_sim_1, out_sim_2], dim=1)
            weights = self.weight_net(concat_features)
            
            fused = (weights[:, 0:1] * out_sim + 
                    weights[:, 1:2] * out_sim_1 + 
                    weights[:, 2:3] * out_sim_2)
        elif len(outputs) == 2:
            out_sim = gamma * outputs[0]
            out_sim_1 = gamma_1 * outputs[1]
            out_sim_2 = gamma_2 * outputs[1]
            
            out_sim = self.transform_1(out_sim)
            out_sim_1 = self.transform_2(out_sim_1)
            out_sim_2 = self.transform_3(out_sim_2)
            
            concat_features = torch.cat([out_sim, out_sim_1, out_sim_2], dim=1)
            weights = self.weight_net(concat_features)
            
            fused = (weights[:, 0:1] * out_sim + 
                    weights[:, 1:2] * out_sim_1 + 
                    weights[:, 2:3] * out_sim_2)
        elif len(outputs) == 1:
            fused = gamma * outputs[0]
        else:
            fused = torch.zeros_like(outputs[0]) if outputs else None
        return fused

class CrossAttentionFusion(BaseFusionModule):
    def __init__(self, d_model, num_heads=8):
        super(CrossAttentionFusion, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.q_proj = nn.Conv2d(d_model, d_model, 1)
        self.k_proj = nn.Conv2d(d_model, d_model, 1)
        self.v_proj = nn.Conv2d(d_model, d_model, 1)
        self.out_proj = nn.Conv2d(d_model, d_model, 1)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, outputs, gamma, gamma_1, gamma_2):
        if len(outputs) == 3:
            B, C, H, W = outputs[0].shape
            
            out_sim = gamma * outputs[0]
            out_sim_1 = gamma_1 * outputs[1]
            out_sim_2 = gamma_2 * outputs[2]
            
            streams = torch.stack([out_sim, out_sim_1, out_sim_2], dim=1)
            streams_flat = streams.view(B, 3, C, H * W)
            
            Q = self.q_proj(out_sim).view(B, self.num_heads, self.d_k, H * W).transpose(2, 3)
            K = self.k_proj(streams_flat.view(B * 3, C, H, W)).view(B, 3, self.num_heads, self.d_k, H * W)
            V = self.v_proj(streams_flat.view(B * 3, C, H, W)).view(B, 3, self.num_heads, self.d_k, H * W)
            
            K = K.permute(0, 2, 1, 4, 3).contiguous().view(B, self.num_heads, 3 * H * W, self.d_k)
            V = V.permute(0, 2, 1, 4, 3).contiguous().view(B, self.num_heads, 3 * H * W, self.d_k)
            
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            attn = self.softmax(scores)
            
            out = torch.matmul(attn, V)
            out = out.transpose(2, 3).contiguous().view(B, C, H, W)
            fused = self.out_proj(out)
        elif len(outputs) == 2:
            B, C, H, W = outputs[0].shape
            
            out_sim = gamma * outputs[0]
            out_sim_1 = gamma_1 * outputs[1]
            out_sim_2 = gamma_2 * outputs[1]
            
            streams = torch.stack([out_sim, out_sim_1, out_sim_2], dim=1)
            streams_flat = streams.view(B, 3, C, H * W)
            
            Q = self.q_proj(out_sim).view(B, self.num_heads, self.d_k, H * W).transpose(2, 3)
            K = self.k_proj(streams_flat.view(B * 3, C, H, W)).view(B, 3, self.num_heads, self.d_k, H * W)
            V = self.v_proj(streams_flat.view(B * 3, C, H, W)).view(B, 3, self.num_heads, self.d_k, H * W)
            
            K = K.permute(0, 2, 1, 4, 3).contiguous().view(B, self.num_heads, 3 * H * W, self.d_k)
            V = V.permute(0, 2, 1, 4, 3).contiguous().view(B, self.num_heads, 3 * H * W, self.d_k)
            
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            attn = self.softmax(scores)
            
            out = torch.matmul(attn, V)
            out = out.transpose(2, 3).contiguous().view(B, C, H, W)
            fused = self.out_proj(out)
        elif len(outputs) == 1:
            fused = gamma * outputs[0]
        else:
            fused = torch.zeros_like(outputs[0]) if outputs else None
        return fused

class SimpleWeightedFusion(BaseFusionModule):
    def __init__(self, d_model):
        super(SimpleWeightedFusion, self).__init__()
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, outputs, gamma, gamma_1, gamma_2):
        if len(outputs) == 3:
            out_sim = gamma * outputs[0]
            out_sim_1 = gamma_1 * outputs[1]
            out_sim_2 = gamma_2 * outputs[2]
            
            weights = self.softmax(self.fusion_weights)
            fused = weights[0] * out_sim + weights[1] * out_sim_1 + weights[2] * out_sim_2
        elif len(outputs) == 2:
            out_sim = gamma * outputs[0]
            out_sim_1 = gamma_1 * outputs[1]
            out_sim_2 = gamma_2 * outputs[1]
            
            weights = self.softmax(self.fusion_weights)
            fused = weights[0] * out_sim + weights[1] * out_sim_1 + weights[2] * out_sim_2
        elif len(outputs) == 1:
            fused = gamma * outputs[0]
        else:
            fused = torch.zeros_like(outputs[0]) if outputs else None
        return fused

