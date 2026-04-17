import os
import math
import warnings
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from model.multiscale_segformer import MultiScaleSegFormer
from transformers.utils import logging as hf_logging

warnings.filterwarnings(
    "ignore",
    message="`resume_download` is deprecated",
    category=FutureWarning,
    module="huggingface_hub.file_download"
)

warnings.filterwarnings(
    "ignore",
    message=".*resume_download.*",
    category=FutureWarning,
)

warnings.filterwarnings("ignore", category=UserWarning, message=".*TypedStorage.*")

hf_logging.set_verbosity_error()

torch.backends.cudnn.deterministic = True

transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def _mask_transform(mask):
    m = np.array(mask)
    if m.ndim == 3:
        m = m[..., 0]
    m = (m > 0).astype(np.int32)
    return m

def masks_transform(masks, numpy=False):
    targets = [_mask_transform(m) for m in masks]
    if numpy:
        return targets
    else:
        return torch.stack([torch.from_numpy(t).long() for t in targets]).cuda()

def images_transform(images):
    inputs = [transformer(img) for img in images]
    return torch.stack(inputs, dim=0).cuda()

def get_patch_info(shape, p_size, overlap_percentage=0.30):
    x, y = shape[0], shape[1]
    n = m = 1
    min_overlap = p_size * overlap_percentage
    while x > n * p_size:
        n += 1
    while p_size - 1.0 * (x - p_size) / (n - 1) < min_overlap:
        n += 1
    while y > m * p_size:
        m += 1
    while p_size - 1.0 * (y - p_size) / (m - 1) < min_overlap:
        m += 1
    return n, m, (x - p_size) * 1.0 / (n - 1), (y - p_size) * 1.0 / (m - 1)

def _get_tissue_mask(image_pil, use_otsu=True, bg_threshold=220):
    image_np = np.array(image_pil.convert("RGB"))
    image_hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    v_channel = image_hsv[:, :, 2]
    
    if use_otsu:
        threshold_value, binary_mask = cv2.threshold(v_channel, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary_mask.astype(np.uint8)
    else:
        return (v_channel < bg_threshold).astype(np.uint8)

def global_to_patch(images, p_size, bg_threshold=220, tissue_coverage_min=0.1, labels=None, overlap_percentage=0.30):
    patches, label_patches, coordinates, templates, sizes = [], [], [], [], []
    ratios = [(0, 0)] * len(images)
    patch_area = p_size[0] * p_size[1]
    patch_ones = np.ones(p_size)
    
    for i in range(len(images)):
        w, h = images[i].size
        size = (h, w)
        sizes.append(size)
        ratios[i] = (float(p_size[0]) / size[0], float(p_size[1]) / size[1])
        
        tissue_mask = _get_tissue_mask(images[i], use_otsu=False, bg_threshold=bg_threshold)
        
        current_patches, current_label_patches, current_coordinates = [], [], []
        template = np.zeros(size)
        n_x, n_y, step_x, step_y = get_patch_info(size, p_size[0], overlap_percentage)
        
        for x in range(n_x):
            top = int(np.round(x * step_x)) if x < n_x - 1 else size[0] - p_size[0]
            
            for y in range(n_y):
                left = int(np.round(y * step_y)) if y < n_y - 1 else size[1] - p_size[1]
                
                if top + p_size[0] > size[0] or left + p_size[1] > size[1]:
                    continue
                
                patch_mask = tissue_mask[top:top+p_size[0], left:left+p_size[1]]
                tissue_coverage = np.sum(patch_mask) / patch_area
                
                if tissue_coverage >= tissue_coverage_min:
                    template[top:top+p_size[0], left:left+p_size[1]] += patch_ones
                    current_coordinates.append((1.0 * top / size[0], 1.0 * left / size[1]))
                    current_patches.append(transforms.functional.crop(images[i], top, left, p_size[0], p_size[1]))
                    if labels is not None:
                        current_label_patches.append(transforms.functional.crop(labels[i], top, left, p_size[0], p_size[1]))
        
        patches.append(current_patches)
        coordinates.append(current_coordinates)
        templates.append(torch.Tensor(template).expand(1, 1, -1, -1).cuda())

        # Validation for sparse images
        if len(current_patches) == 0:
            import warnings
            warnings.warn(
                f"Image {i} produced zero patches after tissue filtering "
                f"(tissue_coverage_min={tissue_coverage_min}). "
                f"Consider reducing tissue_coverage_min or checking image quality."
            )
            h, w = size
            center_top = max(0, (h - p_size[0]) // 2)
            center_left = max(0, (w - p_size[1]) // 2)
            current_patches.append(transforms.functional.crop(
                images[i], center_top, center_left, p_size[0], p_size[1]
            ))
            current_coordinates.append((float(center_top) / h, float(center_left) / w))
            if labels is not None:
                current_label_patches.append(transforms.functional.crop(
                    labels[i], center_top, center_left, p_size[0], p_size[1]
                ))

        if labels is not None:
            label_patches.append(current_label_patches)
    
    if labels is not None:
        return patches, label_patches, coordinates, templates, sizes, ratios
    else:
        return patches, coordinates, templates, sizes, ratios

def global_to_context_patches(images, p_size, patch_coordinates, mul=2):
    P_context_H = int(p_size[0] * mul)
    P_context_W = int(p_size[1] * mul)
    offset_H = int(math.ceil((P_context_H - p_size[0]) / 2.0))
    offset_W = int(math.ceil((P_context_W - p_size[1]) / 2.0))
    patches = []
    
    for i in range(len(images)):
        w, h = images[i].size
        img_np = np.array(images[i])
        size_H, size_W = h, w
        current_context_patches = []
        
        for coord in patch_coordinates[i]:
            top_ratio, left_ratio = coord
            top = int(np.round(top_ratio * size_H))
            left = int(np.round(left_ratio * size_W))
            
            top_start = max(0, top - offset_H)
            left_start = max(0, left - offset_W)
            top_end = min(size_H, top_start + P_context_H)
            left_end = min(size_W, left_start + P_context_W)
            
            large_region = img_np[top_start:top_end, left_start:left_end]
            downsampled = large_region[::mul, ::mul]
            
            if downsampled.shape[0] < p_size[0] or downsampled.shape[1] < p_size[1]:
                padded = np.zeros((p_size[0], p_size[1], 3), dtype=np.uint8)
                h_actual, w_actual = downsampled.shape[:2]
                padded[:h_actual, :w_actual] = downsampled
                downsampled = padded
            else:
                downsampled = downsampled[:p_size[0], :p_size[1]]
            
            context_patch = Image.fromarray(downsampled)
            current_context_patches.append(context_patch)
        patches.append(current_context_patches)
    return patches

def stitch_patch_predictions_to_global(patches, n_class, sizes, coordinates, p_size, templates=None):
    predictions = [np.zeros((n_class, size[0], size[1])) for size in sizes]
    for i in range(len(sizes)):
        for j in range(len(coordinates[i])):
            top = int(np.round(coordinates[i][j][0] * sizes[i][0]))
            left = int(np.round(coordinates[i][j][1] * sizes[i][1]))
            predictions[i][:, top:top+p_size[0], left:left+p_size[1]] += patches[i][j]
    
    if templates is not None:
        for i in range(len(sizes)):
            overlap_counts = templates[i]
            if hasattr(overlap_counts, "cpu"):
                overlap_counts = overlap_counts.cpu().numpy()
            overlap_counts = np.asarray(overlap_counts).squeeze()
            predictions[i] /= (overlap_counts + 1e-8)
    return predictions

def collate(batch):
    image = [b['image'] for b in batch]
    label = [b['label'] for b in batch]
    id = [b['id'] for b in batch]
    img_name = [b.get('img_name', None) for b in batch]
    return {'image': image, 'label': label, 'id': id, 'img_name': img_name}

def collate_test(batch):
    image = [b['image'] for b in batch]
    id = [b['id'] for b in batch]
    img_name = [b.get('img_name', None) for b in batch]
    return {'image': image, 'id': id, 'img_name': img_name}

def _init_dmmn_weights(model):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

def _print_model_params(model, model_name="Model"):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{model_name}] Total: {total:,} | Trainable: {trainable:,} | Frozen: {total-trainable:,}")

def create_model_load_weights(n_class, pre_path="", input_mode=3, use_window=False):
    def _maybe_partial_load(model, ckpt_path):
        if not (ckpt_path and os.path.isfile(ckpt_path)):
            if ckpt_path:
                print(f"[warn] skip load (missing file): {ckpt_path}")
            return 0
        
        print(f"[info] loading: {ckpt_path}")
        try:
            blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            blob = torch.load(ckpt_path, map_location="cpu")
        
        state = None
        if isinstance(blob, dict):
            for k in ("state_dict", "model_state", "model", "net", "params"):
                if k in blob and isinstance(blob[k], dict):
                    state = blob[k]
                    break
        if state is None:
            state = blob if isinstance(blob, dict) else {}
        
        def unprefix(d):
            out = {}
            for k, v in d.items():
                if k.startswith("module."):
                    out[k[len("module."):]] = v
                else:
                    out[k] = v
            return out
        
        src_un = unprefix(state)
        tgt = model.state_dict()
        tgt_un = unprefix(tgt)
        
        to_load_un = {k: v for k, v in src_un.items() if k in tgt_un and tgt_un[k].shape == v.shape}
        
        remapped = {}
        loaded, total = 0, len(to_load_un)
        for k, v in to_load_un.items():
            k_target = k if k in tgt else ("module." + k if ("module." + k) in tgt else None)
            if k_target is not None:
                remapped[k_target] = v
                loaded += 1
        
        if loaded == 0:
            print("[warn] no matching keys to load (check encoder/arch and DP prefix).")
            return 0
        
        tgt.update(remapped)
        model.load_state_dict(tgt, strict=False)
        print(f"[info] loaded {loaded}/{total} matched keys.")
        return loaded
    
    model = None
    if input_mode in (1, 2, 3):
        print(f'Loading Multi-Scale SegFormer - input_mode={input_mode}, use_window={use_window}...')
        model = MultiScaleSegFormer(
            n_class=n_class,
            variant="b0",
            pretrained=True,
            share_encoder=False, 
            input_mode=input_mode,
            use_window=use_window,
        ).cuda()
        
        if pre_path:
            _maybe_partial_load(model, pre_path)
        model = nn.DataParallel(model).cuda()
        _print_model_params(model, "Multi-Scale Segformer")
            
    return model