# SAMHA: Spatial-Aware Multi-Head Attention for Medical Image Segmentation

SAMHA is a PyTorch framework for binary medical image segmentation. It combines
multi-scale image context with spatial-aware attention so the model can use both
local tissue detail and wider anatomical context during prediction.s

## Overview

The model uses a multi-field-of-view (multi-FOV) pipeline. Local, medium, and
large image patches are encoded in parallel, fused with SAMHA attention, and
decoded into a final segmentation mask.

![SAMHA overall workflow](assets/architecture/multi-fov-arcitecture.png)

SAMHA includes two attention variants:

- **SAMHA**: distance-aware cross-scale attention with learnable fusion.
- **SAMHA-Window**: window-based attention for efficient high-resolution stages.

![SAMHA and SAMHA-Window modules](assets/architecture/samha-samhwin.png)

In the module figure, **A** denotes the SAMHA block and **B** denotes the
SAMHA-Window block. The deepest scale (H4) always uses full SAMHA attention,
since its feature map is small. H1-H3 are much higher-resolution feature
maps, where full (quadratic-cost) SAMHA attention is computationally
expensive and impractical — so those scales should use the cheaper, windowed
SAMHA-Window attention instead. **`--use_window true` is the default and
should be left on whenever `--input_mode` is `2` or `3`** (see
[Training Options](#training-options)). Setting it `false` still runs, but
falls back to full (high-attention) SAMHA on H1-H3, which is much slower and
more memory-hungry — only do this deliberately, e.g. for a small-scale
comparison/ablation.

## Repository Structure

```text
SAMHA/
|-- train.py                     # Training entry point
|-- trainer.py                   # Training and evaluation loops
|-- args.py                      # Command-line arguments
|-- requirements.txt             # Python dependencies
|-- assets/architecture/         # Architecture figures
|-- dataset/dataloader.py        # Dataset loading
|-- model/                       # Model and attention modules
|-- notebook/run_samha.ipynb     # Single-image inference demo
`-- utils/                       # Metrics, losses, schedulers, inference helpers
```

Generated files such as checkpoints, TensorBoard logs, and notebook predictions
are written under `saved_models/`, `runs/`, or `notebook/` depending on the
workflow.

## Installation

Create and activate a Python environment, then install the project
dependencies:

```bash
pip install -r requirements.txt
```

The project uses PyTorch, torchvision, NumPy, Pillow, OpenCV, scikit-learn,
TensorBoard, Weights & Biases, and OpenSlide for whole-slide image support.

## Dataset Layout

The training script currently maps `--dataset 1` to `../dataset/dataset1/` and
`--dataset 2` to `../dataset/dataset2/`. Each dataset should follow this
structure:

```text
dataset1/
|-- train/
|   |-- images/
|   `-- gt/
`-- test/
    |-- images/
    `-- gt/
```

For `--dataset 1`, mask filenames should match image filenames. For
`--dataset 2`, masks are expected in `gt/` with the pattern
`<image_name>_mask.<ext>`.

## Quick Start

Run training and validation with all three context scales:

```bash
python train.py \
    --dataset 1 \
    --input_mode 3 \
    --use_window true \
    --task_name samha_run \
    --experiment exp01 \
    --train \
    --val
```

Run the notebook demo for single-image inference:

```bash
jupyter notebook notebook/run_samha.ipynb
```

Set the dataset path, checkpoint path, and image name inside the notebook before
running all cells.

## Training Options

Common arguments:

```bash
python train.py \
    --dataset 1 \
    --input_mode 3 \
    --num_epochs 50 \
    --batch_size 3 \
    --size_p 672 \
    --size_g 672 \
    --context_M 2 \
    --context_L 3 \
    --patch_overlap 0.20 \
    --task_name "samha_run" \
    --experiment "exp01" \
    --train \
    --val
```

Important options:

| Argument             | Description                                                                                                                                                                                                                                                                                                    |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--dataset`        | `1` for patch images, `2` for whole-slide image loading                                                                                                                                                                                                                                                    |
| `--input_mode`     | `1` local only, `2` local + medium, `3` local + medium + large                                                                                                                                                                                                                                           |
| `--use_window`     | Use windowed (low-cost) SAMHA-Window attention for the H1-H3 scales instead of full SAMHA. Default`true`. Leave `true` whenever `--input_mode` is `2` or `3` — setting `false` still runs, but is much slower/more memory-hungry (full quadratic-cost attention on high-resolution feature maps). |
| `--num_epochs`     | Number of training epochs                                                                                                                                                                                                                                                                                      |
| `--batch_size`     | Batch size for global images                                                                                                                                                                                                                                                                                   |
| `--sub_batch_size` | Batch size for local patch processing                                                                                                                                                                                                                                                                          |
| `--size_p`         | Local patch size                                                                                                                                                                                                                                                                                               |
| `--size_g`         | Global image resize size                                                                                                                                                                                                                                                                                       |
| `--context_M`      | Medium context multiplier                                                                                                                                                                                                                                                                                      |
| `--context_L`      | Large context multiplier                                                                                                                                                                                                                                                                                       |
| `--patch_overlap`  | Overlap ratio used during patch inference                                                                                                                                                                                                                                                                      |
| `--gpu`            | GPU device ID, for example`0` or `0,1`                                                                                                                                                                                                                                                                     |

## Multi-Scale Modes

SAMHA can use one, two, or three input scales:

| Mode  | Inputs                 | Use Case                                |
| ----- | ---------------------- | --------------------------------------- |
| `1` | Local                  | Faster inference with local detail only |
| `2` | Local + medium         | Adds regional context                   |
| `3` | Local + medium + large | Uses the full multi-scale design        |

The local patch size is controlled by `--size_p`. The medium and large context
sizes are derived from `--context_M` and `--context_L`.

## How SAMHA Works

1. **Multi-scale input**: local, medium, and large patches are prepared from the
   same image region.
2. **Parallel encoding**: each scale is processed through its own encoder
   stream.
3. **Distance-aware attention**: SAMHA adds a feature-grid proximity bias to
   the attention logits so nearby resized-grid tokens are favoured while still
   allowing long-range interactions when content is similar.
4. **Learnable fusion**: features from each scale are combined with learned
   weights.
5. **Decoding**: fused features are upsampled into a segmentation mask.

### Distance-prior ablations

The main model is **exponential learned**:

\[
B_{ij}=\lambda_{\mathrm{dist}}
\exp\left(-D_{ij}^{\mathrm{grid}}/\sigma\right),
\]

where \(D_{ij}^{\mathrm{grid}}\) is Euclidean distance in feature-token units,
\(\sigma\) is fixed, and \(\lambda_{\mathrm{dist}}\) is learned from an initial
value of 0.1. The code intentionally supports only five effective ablations:

| Variant                    | `--distance_prior` | `--lambda_dist_trainable` |
| -------------------------- | -------------------- | --------------------------- |
| Exponential learned (main) | `exp`              | `True`                    |
| Exponential fixed          | `exp`              | `False`                   |
| Gaussian learned           | `gaussian`         | `True`                    |
| Gaussian fixed             | `gaussian`         | `False`                   |
| No distance bias           | `none`             | forced to`False`          |

The Gaussian kernel is
\(\exp[-(D_{ij}^{\mathrm{grid}})^2/(2\sigma^2)]\). The `none` control removes
the additive distance bias but retains the positional encoding. These distances
are computed on the common resized feature grid and are not physical WSI
distances.

## Outputs

Training checkpoints are saved to:

```text
saved_models/{dataset}/{experiment}/
```

TensorBoard logs are saved to:

```text
runs/{dataset}/{experiment}/
```

View logs with:

```bash
tensorboard --logdir=./runs
```

The notebook demo saves prediction arrays and mask visualizations in the
notebook output directory.

## Troubleshooting

| Problem                | Suggested Fix                                                                              |
| ---------------------- | ------------------------------------------------------------------------------------------ |
| CUDA out of memory     | Reduce`--batch_size`, `--sub_batch_size`, or `--input_mode`                          |
| Dataset path error     | Check the expected`train/images`, `train/gt`, `test/images`, and `test/gt` folders |
| Mask not found         | Confirm mask filenames match the expected naming rule                                      |
| OpenSlide import error | Install OpenSlide system libraries and`openslide-python`                                 |
| GPU not selected       | Set`--gpu`, for example `--gpu 0`                                                      |
