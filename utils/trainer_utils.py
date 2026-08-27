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

def global_to_patch(images, p_size, bg_threshold=220, tissue_coverage_min=0.1, labels=None, overlap_percentage=0.20):
    patches, label_patches, coordinates, templates, sizes = [], [], [], [], []
    ratios = [(0, 0)] * len(images)
    patch_area = p_size[0] * p_size[1]
    patch_ones = np.ones(p_size)
    
    for i in range(len(images)):
        w, h = images[i].size
        size = (h, w)
        sizes.append(size)
        ratios[i] = (float(p_size[0]) / size[0], float(p_size[1]) / size[1])
        
        tissue_mask = _get_tissue_mask(images[i], use_otsu=True, bg_threshold=bg_threshold)
        
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

def stitch_patch_predictions_to_global(patches, n_class, sizes, coordinates, p_size, templates=None, 
                                       use_blend=True, blend_method='linear', edge_fade=0.04):
    predictions = [np.zeros((n_class, size[0], size[1])) for size in sizes]
    if use_blend:
        weight_sums = [np.zeros((size[0], size[1])) for size in sizes]
        weight_mask = _create_weight_mask(p_size, method=blend_method, edge_fade=edge_fade)
        
        for i in range(len(sizes)):
            for j in range(len(coordinates[i])):
                top = int(np.round(coordinates[i][j][0] * sizes[i][0]))
                left = int(np.round(coordinates[i][j][1] * sizes[i][1]))
                
                for c in range(n_class):
                    predictions[i][c, top:top+p_size[0], left:left+p_size[1]] += patches[i][j][c] * weight_mask
                weight_sums[i][top:top+p_size[0], left:left+p_size[1]] += weight_mask
        
        for i in range(len(sizes)):
            for c in range(n_class):
                predictions[i][c] /= (weight_sums[i] + 1e-8)
    else:
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

def _create_weight_mask(patch_size, method='linear', sigma_scale=0.25, edge_fade=0.10):
    """
    Create blending weight mask. Methods:
    'linear': Fades at edges only - BEST for medical (preserves boundaries, removes seams)
    'cosine': Raised cosine - good balance
    'gaussian': Smooth everywhere - may blur boundaries
    """
    h, w = patch_size
    
    if method == 'gaussian':
        center_h, center_w = h // 2, w // 2
        sigma_h, sigma_w = h * sigma_scale, w * sigma_scale
        y, x = np.ogrid[:h, :w]
        gaussian_h = np.exp(-((y - center_h) ** 2) / (2 * sigma_h ** 2))
        gaussian_w = np.exp(-((x - center_w) ** 2) / (2 * sigma_w ** 2))
        weight_mask = gaussian_h * gaussian_w
        weight_mask = (weight_mask - weight_mask.min()) / (weight_mask.max() - weight_mask.min() + 1e-8)
        
    elif method == 'cosine':
        fade_h = int(h * edge_fade)
        fade_w = int(w * edge_fade)
        weight_y = np.ones(h)
        weight_x = np.ones(w)
        if fade_h > 0:
            weight_y[:fade_h] = 0.5 * (1 - np.cos(np.pi * np.arange(fade_h) / fade_h))
            weight_y[-fade_h:] = 0.5 * (1 - np.cos(np.pi * np.arange(fade_h, 0, -1) / fade_h))
        if fade_w > 0:
            weight_x[:fade_w] = 0.5 * (1 - np.cos(np.pi * np.arange(fade_w) / fade_w))
            weight_x[-fade_w:] = 0.5 * (1 - np.cos(np.pi * np.arange(fade_w, 0, -1) / fade_w))
        weight_mask = weight_y[:, None] * weight_x[None, :]
        
    else: 
        fade_h = int(h * edge_fade)
        fade_w = int(w * edge_fade)
        weight_y = np.ones(h)
        weight_x = np.ones(w)
        if fade_h > 0:
            weight_y[:fade_h] = np.linspace(0, 1, fade_h)
            weight_y[-fade_h:] = np.linspace(1, 0, fade_h)
        if fade_w > 0:
            weight_x[:fade_w] = np.linspace(0, 1, fade_w)
            weight_x[-fade_w:] = np.linspace(1, 0, fade_w)
        weight_mask = weight_y[:, None] * weight_x[None, :]
    
    return weight_mask

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

def _print_model_params(model, model_name="Model"):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{model_name}] Total: {total:,} | Trainable: {trainable:,} | Frozen: {total-trainable:,}")

def normalize_distance_prior_config(
    distance_prior,
    distance_sigma,
    lambda_dist_init,
    lambda_dist_trainable,
):
    """Validate and canonicalize the five supported distance ablations."""
    if distance_prior is None:
        distance_prior = "none"
    allowed_priors = {"exp", "gaussian", "none"}
    if distance_prior not in allowed_priors:
        raise ValueError(
            f"Unknown distance_prior: {distance_prior}. "
            f"Expected one of {sorted(allowed_priors)}."
        )
    if not math.isfinite(distance_sigma) or distance_sigma <= 0:
        raise ValueError(
            f"distance_sigma must be a finite positive value, got {distance_sigma}."
        )
    if not math.isfinite(lambda_dist_init):
        raise ValueError(
            f"lambda_dist_init must be finite, got {lambda_dist_init}."
        )

    if distance_prior == "none":
        lambda_dist_init = 0.0
        lambda_dist_trainable = False
        distance_variant = "none"
    else:
        distance_variant = (
            f"{distance_prior}-"
            f"{'learned' if lambda_dist_trainable else 'fixed'}"
        )

    return (
        distance_prior,
        distance_sigma,
        lambda_dist_init,
        lambda_dist_trainable,
        distance_variant,
    )

def create_model_load_weights(
    n_class,
    pre_path="",
    input_mode=3,
    use_window=False,
    distance_prior="exp",
    distance_sigma=1.0,
    lambda_dist_init=0.1,
    lambda_dist_trainable=True,
):
    (
        distance_prior,
        distance_sigma,
        lambda_dist_init,
        lambda_dist_trainable,
        _,
    ) = normalize_distance_prior_config(
        distance_prior,
        distance_sigma,
        lambda_dist_init,
        lambda_dist_trainable,
    )

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

        requested_config = {
            "input_mode": input_mode,
            "use_window": use_window,
            "distance_prior": distance_prior,
            "distance_sigma": distance_sigma,
            "lambda_dist_init": lambda_dist_init,
            "lambda_dist_trainable": lambda_dist_trainable,
            "distance_variant": (
                "none"
                if distance_prior == "none"
                else f"{distance_prior}-{'learned' if lambda_dist_trainable else 'fixed'}"
            ),
        }
        saved_config = blob.get("model_config") if isinstance(blob, dict) else None
        if isinstance(saved_config, dict):
            mismatches = []
            for key, requested_value in requested_config.items():
                if key not in saved_config:
                    continue
                saved_value = saved_config[key]
                if isinstance(requested_value, float):
                    try:
                        values_match = math.isclose(
                            float(saved_value),
                            requested_value,
                            rel_tol=1e-9,
                            abs_tol=1e-12,
                        )
                    except (TypeError, ValueError):
                        values_match = False
                else:
                    values_match = saved_value == requested_value
                if not values_match:
                    mismatches.append(
                        f"{key}: checkpoint={saved_value!r}, requested={requested_value!r}"
                    )
            if mismatches:
                warnings.warn(
                    "Checkpoint configuration differs from the requested model: "
                    + "; ".join(mismatches),
                    RuntimeWarning,
                )
        else:
            warnings.warn(
                "Checkpoint has no model_config metadata; its distance-prior "
                "variant cannot be verified.",
                RuntimeWarning,
            )
        
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
            distance_prior=distance_prior,
            distance_sigma=distance_sigma,
            lambda_dist_init=lambda_dist_init,
            lambda_dist_trainable=lambda_dist_trainable,
        ).cuda()
        
        if pre_path:
            _maybe_partial_load(model, pre_path)
        model = nn.DataParallel(model).cuda()
        _print_model_params(model, "Multi-Scale Segformer")
            
    return model
