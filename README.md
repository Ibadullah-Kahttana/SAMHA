# SAMHA: Spatial-Aware Multi-Head Attention for Medical Image Segmentation

A deep learning framework for segmenting tumors in medical images using multi-scale analysis and spatial attention.

## What is SAMHA?

SAMHA addresses tumor segmentation in histopathology images by processing patches at **three concurrent scales**:

1. **Local patches** (672×672) - Detailed tissue features
2. **Medium context** (1344×1344) - Neighboring tissue information  
3. **Large context** (2016×2016) - Broader anatomical context

The key innovation is **Spatial-Aware Multi-Head Attention**: instead of treating all tissue regions equally, SAMHA learns to pay more attention to **spatially nearby tissue** while still allowing long-range connections when needed. This is done by injecting a distance penalty into the attention mechanism—closer regions get higher weights, distant regions get lower weights.

The framework then **fuses these three scales** using learned weights, automatically deciding which scale to trust most at each location. This is especially useful for **boundary detection** where accurate tumor-stroma interfaces require both local detail and broader context.

**Result**: Better segmentation accuracy (higher overlap with ground truth) and stronger boundary performance, even when staining varies across images.

## Quick Start (5 minutes)

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Test on One Image (Jupyter Notebook)
```bash
jupyter notebook notebook/run_samha.ipynb
```
Edit Cell 3 with your dataset path, then run all cells. Predictions save automatically.

### 3. Train on Your Data
```bash
python train.py --dataset 1 --input_mode 3 --train --val
```

```
SAMHA/
├── train.py                    # Main training script
├── trainer.py                  # Training & evaluation classes
├── args.py                     # Command-line arguments
├── requirements.txt            # Dependencies
├── README.md                   # This file
│
├── model/
│   ├── SAMHA.py               # SAMHA attention module
│   ├── multiscale_segformer.py # Model architecture
│   ├── attention_modules.py    # Implementation details
│   └── [other modules]
│
├── dataset/
│   └── dataloader.py           # Data loading
│
├── utils/
│   ├── trainer_utils.py        # Patch stitching, inference
│   ├── loss.py                 # Loss functions
│   └── lr_scheduler.py         # Learning rate scheduling
│
├── notebook/
│   ├── run_samha.ipynb         # Single-image demo (recommended)
│   └── [saved predictions]
│
├── saved_models/
│   └── [trained checkpoints]
│
└── runs/
    └── [TensorBoard logs]
```

## Understanding the Code

### Key Files Explained

**`train.py`** - Entry point for training
- Parses arguments from `args.py`
- Loads dataset using `dataset/dataloader.py`
- Creates model using `model/` modules
- Runs training loop with `trainer.py`
- Saves checkpoints to `saved_models/`

**`model/multiscale_segformer.py`** - Main model architecture
- Takes 1-3 input patches (local, medium, large)
- Encodes each scale separately
- Fuses features using SAMHA attention
- Decodes to segmentation output

**`model/SAMHA.py`** - Spatial-Aware Multi-Head Attention
- **Projection**: Projects all 3 inputs to same dimension
- **Distance Prior**: Creates spatial distance matrix (closer = higher attention)
- **Attention**: Computes attention with distance bias
- **Fusion**: Blends 3 scales with learned weights

**`trainer.py`** - Training & evaluation
- `Trainer` class: Trains for one epoch
- `Evaluator` class: Tests on validation/test set
- Computes metrics (IoU, Dice, Hausdorff Distance)

**`utils/trainer_utils.py`** - Inference helpers
- `global_to_patch()` - Splits large image into overlapping patches
- `stitch_patch_predictions_to_global()` - Stitches predictions back to full image size
- `create_model_load_weights()` - Loads model with checkpoint

**`dataset/dataloader.py`** - Data loading
- Dataset 1: Dataset1 (structured patches)
- Dataset 2: Dataset2 (OpenSlide WSI format)
- Auto-detects label folders (gt/, masks/, mask/)

## Training Configuration

Essential arguments:
```bash
python train.py \
    --dataset 1              # 1=Dataset1, 2=Dataset2
    --input_mode 3           # 1=Local, 2=Local+Medium, 3=All three
    --num_epochs 100         # Training epochs
    --batch_size 4           # Batch size per GPU
    --size_p 672             # Patch size (672×672)
    --context_M 2            # Medium context multiplier (×2 = 1344×1344)
    --context_L 3            # Large context multiplier (×3 = 2016×2016)
    --train --val            # Training and validation
```

**Examples:**
```bash
# Dataset1 (local + all contexts)
python train.py --dataset 1 --input_mode 3 --train --val

# Dataset2 WSI
python train.py --dataset 2 --input_mode 3 --train --val
```

## Datasets

- **Dataset 1**: Dataset1 histopathology (tissue patches, binary segmentation)
- **Dataset 2**: Dataset2 (whole slide images, WSI format)

## Jupyter Notebook Demo

The easiest way to test SAMHA is using `notebook/run_samha.ipynb`:

1. **Load test image** - Pick one image from your dataset
2. **Load model** - Load trained weights from checkpoint
3. **Run inference** - Process image through SAMHA
4. **Save results** - Predictions saved as `.npy` and `.png`

**What you specify:**
- `DATA_ROOT` - Path to your dataset folder
- `CHECKPOINT_PATH` - Path to trained model weights
- `IMAGE_NAME` - Which image to test (auto-picks first if not set)

**What you get:**
- `.npy` file - Raw prediction array for analysis
- `.png` file - Visualization of prediction mask
- Matplotlib plot - Instant visual feedback

## Multi-Scale Processing

SAMHA processes three patch sizes simultaneously:

| Mode | Local | Medium | Large | Best For |
|------|-------|--------|-------|----------|
| **1** | 672×672 | No | No | Fast inference |
| **2** | 672×672 | 1344×1344 | No | Balanced |
| **3** | 672×672 | 1344×1344 | 2016×2016 | Best accuracy |

- **Local**: Fine tissue details
- **Medium**: Regional context (2× multiplier)
- **Large**: Global anatomical context (3× multiplier)

The model learns to blend all three scales intelligently.

## How SAMHA Works (Step-by-Step)

### 1. Input: Three Patch Scales
```
Tissue Image
    ├── Local patch (672×672)     → X_local
    ├── Medium patch (1344×1344)  → X_medium  
    └── Large patch (2016×2016)   → X_large
```

### 2. Encoding: Separate Encoders
Each scale is processed through its own encoder stream to extract features.

### 3. SAMHA Attention (The Key Innovation)

**Step 3a - Projection:**
All features are projected to the same embedding dimension so they can be compared.

**Step 3b - Distance Prior:**
A spatial distance matrix `D` is created based on pixel coordinates:
- Close pixels (same tissue region) → high weight (close to 1)
- Distant pixels (far apart) → low weight (close to 0)
- Formula: `D = exp(-distance / σ)` (exponential decay)

**Step 3c - Distance-Aware Attention:**
Instead of standard attention, the logits include a distance penalty:
```
Attention = softmax(content_similarity + λ × log(distance_matrix))
```
This means:
- Nearby tissue regions get higher attention weights
- Distant regions get lower weights
- But if content is very similar distant regions can still be attended to

**Step 3d - Three Streams Fused:**
Outputs from all three scales are weighted:
```
Output = α₁×output_local + α₂×output_medium + α₃×output_large
```
where α₁, α₂, α₃ are learned weights (summing to 1) that the model learns to adjust.

### 4. Decoding: Back to Full Image
The fused features are upsampled back to the original image resolution.

## Results

Training outputs go to:
- **Models**: `saved_models/{dataset}/{experiment}/`
- **Logs**: `runs/{dataset}/{experiment}/` (view with TensorBoard)

Inference outputs go to:
- **Notebook**: `notebook/{image_name}_pred_mask.npy` and `.png`

View training progress:
```bash
tensorboard --logdir=./runs
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| CUDA out of memory | Reduce `--batch_size` |
| Slow inference | Use `--input_mode 1` or `2` |
| Data not found | Check dataset folder names (`gt/`, `masks/`, etc.) |
| GPU not detected | Check `GPU_DEVICES` in notebook Cell 2 |

## Dependencies

Core packages: PyTorch, NumPy, Pillow, OpenCV, scikit-learn, TensorBoard, WandB, OpenSlide (for WSI support).

Full list in `requirements.txt`. Install with:
```bash
pip install -r requirements.txt
```

