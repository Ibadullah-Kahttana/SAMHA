import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformers import SegformerConfig, SegformerModel

_BACKBONES = {
    "b0": "nvidia/segformer-b0-finetuned-ade-512-512",
    "b1": "nvidia/segformer-b1-finetuned-ade-512-512",
    "b2": "nvidia/segformer-b2-finetuned-ade-512-512",
    "b3": "nvidia/segformer-b3-finetuned-ade-512-512",
    "b4": "nvidia/segformer-b4-finetuned-ade-512-512",
    "b5": "nvidia/segformer-b5-finetuned-ade-640-640",
}

_ENCODER_BACKBONES = {
    "b0": "nvidia/mit-b0",
    "b1": "nvidia/mit-b1",
    "b2": "nvidia/mit-b2",
    "b3": "nvidia/mit-b3",
    "b4": "nvidia/mit-b4",
    "b5": "nvidia/mit-b5",
}

class MiniSegFormer(nn.Module):
    def __init__(self, variant="b0"):
        super().__init__()
        self.variant = (variant or "b0").lower()
        enc_id = _ENCODER_BACKBONES.get(self.variant, _ENCODER_BACKBONES["b0"])
        cfg = SegformerConfig.from_pretrained(enc_id)
        self.encoder = SegformerModel(cfg)
    
    def forward(self, x: torch.Tensor):
        outputs = self.encoder(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True
        )
        return outputs.hidden_states
