import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from transformers import (
    SegformerConfig,
    SegformerModel,
    SegformerForSemanticSegmentation,
)

from model.attention_modules import SAMHAChannelGate, FuseLocalAndContext
from model.SAMHA import SAMHA, SAMHAWindow
from model.decoders import UNetStyleDecoder, UpsampleRefinement

warnings.filterwarnings(
    "ignore",
    message=".*resume_download.*",
    category=FutureWarning,
)

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


class MultiScaleSegFormer(nn.Module):
    def __init__(
        self,
        n_class: int,
        variant: str = "b0",
        pretrained: bool = True,
        share_encoder: bool = True,
        input_mode: int = 3,
        use_window: bool = False,
        distance_prior: str = 'exp',
        distance_sigma: float = 1.0,
        lambda_dist_init: float = 0.1,
        lambda_dist_trainable: bool = True,
    ):
        super().__init__()
        self.n_class = int(n_class)
        self.variant = (variant or "b0").lower()
        self.share_encoder = share_encoder
        self.input_mode = input_mode

        seg_id = _BACKBONES.get(self.variant, _BACKBONES["b0"])
        enc_id = _ENCODER_BACKBONES.get(self.variant, _ENCODER_BACKBONES["b0"])

        self.base_model = self._build_base_model(seg_id, enc_id, pretrained)
        self._init_decode_head(self.base_model, self.n_class)
        self._init_encoders(self.base_model, enc_id, pretrained, share_encoder)
        
        if self.variant == "b0":
            enc_channels = [32, 64, 160, 256]
        else:
            enc_channels = [64, 128, 320, 512]

        # Hidden Layers H1, H2, H3, H4
        # H4: SAMHA
        self.attention_H4 = SAMHA(
            enc_channels[3], enc_channels[3], num_heads=8, d_model=enc_channels[3], 
            fusion_type='simple_weighted',
            distance_prior=distance_prior,
            distance_sigma=distance_sigma,
            lambda_dist_init=lambda_dist_init,
            lambda_dist_trainable=lambda_dist_trainable
        )

        # H1, H2, H3 : SAMHAWindow or SAMHA
        # SAMHA : High Attention Module
        # SAMHAWindowe : Low Attention Module
        if use_window:
            self.attention_H1 = SAMHAWindow(enc_channels[0], num_heads=4, window_size=8)
            self.attention_H2 = SAMHAWindow(enc_channels[1], num_heads=4, window_size=8)
            self.attention_H3 = SAMHAWindow(enc_channels[2], num_heads=4, window_size=8)
        else:      
            self.attention_H1 = SAMHAChannelGate(enc_channels[0])
            self.attention_H2 = SAMHAChannelGate(enc_channels[1])
            self.attention_H3 = SAMHAChannelGate(enc_channels[2])    

        # UnetStyle Decoder
        self.unet_decoder = UNetStyleDecoder(enc_channels, num_classes=self.n_class)
        self.upsample_refine = UpsampleRefinement(num_classes=self.n_class)

    def _build_base_model(self, seg_id: str, enc_id: str, pretrained: bool):
        num_labels = max(self.n_class, 1)
        if pretrained:
            base = SegformerForSemanticSegmentation.from_pretrained(
                seg_id,
                num_labels=num_labels,
                ignore_mismatched_sizes=True,
            )
        else:
            cfg = SegformerConfig.from_pretrained(enc_id)
            cfg.num_labels = num_labels
            base = SegformerForSemanticSegmentation(cfg)

        return base
    
    def _init_decode_head(self, base_model, n_class: int):
        self.decode_head = base_model.decode_head
        num_labels = max(n_class, 1)
        cls_conv = self.decode_head.classifier
        if cls_conv.out_channels != num_labels:
            self.decode_head.classifier = nn.Conv2d(
                in_channels=cls_conv.in_channels,
                out_channels=num_labels,
                kernel_size=1,
            )

    def _init_encoders(self, base_model, enc_id: str, pretrained: bool, share_encoder: bool):
        self.encoder_local: SegformerModel = base_model.segformer
        if share_encoder:
            self.encoder_medium = self.encoder_local
            self.encoder_large = self.encoder_local
        else:
            if pretrained:
                try:
                    self.encoder_medium = SegformerModel.from_pretrained(enc_id)
                except:
                    full_model_medium = SegformerForSemanticSegmentation.from_pretrained(enc_id, num_labels=150, ignore_mismatched_sizes=True)
                    self.encoder_medium = full_model_medium.segformer
        
                try:
                    self.encoder_large = SegformerModel.from_pretrained(enc_id)
                except:
                    full_model_large = SegformerForSemanticSegmentation.from_pretrained(enc_id, num_labels=150, ignore_mismatched_sizes=True)
                    self.encoder_large = full_model_large.segformer
            else:
                cfg = SegformerConfig.from_pretrained(enc_id)
                self.encoder_medium = SegformerModel(cfg)
                self.encoder_large = SegformerModel(cfg)

    def _encode(self, encoder: SegformerModel, x: torch.Tensor):
        outputs = encoder(pixel_values=x, output_hidden_states=True, return_dict=True)
        return outputs.hidden_states

    # MLP decoder
    def _decode(self, fused_hidden):
        return self.decode_head(fused_hidden)

    def forward(self, x_local, x_medium=None, x_large=None):
        feats_local = self._encode(self.encoder_local, x_local)
        X1, X2, X3, X4 = feats_local

        if self.input_mode == 1:
            fused_hidden = (X1, X2, X3, X4)

        elif self.input_mode == 2:
            if x_medium is None:
                raise ValueError("x_medium required for input_mode=2")
            feats_medium = self._encode(self.encoder_medium, x_medium)
            M1, M2, M3, M4 = feats_medium

            attn_1 = self.attention_H1(X1, M1)
            attn_2 = self.attention_H2(X2, M2)
            attn_3 = self.attention_H3(X3, M3)
            attn_4 = self.attention_H4(x=X4, y=M4)
            fused_hidden = (attn_1, attn_2, attn_3, attn_4)
        
        elif self.input_mode == 3:
            if x_medium is None or x_large is None:
                raise ValueError("x_medium and x_large required for input_mode=3")
            feats_medium = self._encode(self.encoder_medium, x_medium)
            feats_large = self._encode(self.encoder_large, x_large)
        
            M1, M2, M3, M4 = feats_medium
            G1, G2, G3, G4 = feats_large
            
            attn_1 = self.attention_H1(X1, M1, G1)
            attn_2 = self.attention_H2(X2, M2, G2)
            attn_3 = self.attention_H3(X3, M3, G3)
            attn_4 = self.attention_H4(x=X4, y=M4, z=G4)   
            fused_hidden = (attn_1, attn_2, attn_3, attn_4)
        else:
            raise ValueError(f"Invalid input mode: {self.input_mode}")

        logits = self.unet_decoder(fused_hidden) # unet decoder
        #logits = self._decode(fused_hidden) # mlp decoder
    
        if logits.shape[-2:] != x_local.shape[-2:]:
            mask = F.interpolate(logits, size=x_local.shape[-2:], mode="bilinear", align_corners=False)
            mask = self.upsample_refine(mask)
        else:
            mask = logits
        
        return mask
