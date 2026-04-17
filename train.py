from __future__ import absolute_import, division, print_function

import os
import warnings
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import datetime
import time
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

warnings.filterwarnings("ignore", category=UserWarning, message=".*TypedStorage.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*resume_download.*")

from dataset.dataloader import DATALOADER, classToRGB, is_image_file
from utils.loss import (
    FocalLoss,
    FocalDiceComboLoss,
    DiceLoss,
)
from utils.lr_scheduler import get_optimizer_and_scheduler
from utils.trainer_utils import create_model_load_weights, collate
from trainer import Trainer, Evaluator
from args import Args

args = Args().parse()
dataset = args.dataset
use_wandb = args.use_wandb
num_epochs = args.num_epochs
lens = args.lens
start = args.start
task_name = args.task_name
experiment = args.experiment
input_mode = getattr(args, 'input_mode', 3)
train = args.train
val = args.val
n_class = args.n_class
size_p = (args.size_p, args.size_p)
size_g = (args.size_g, args.size_g)
sub_batch_size = args.sub_batch_size
context_M = args.context_M
context_L = args.context_L
patch_overlap = args.patch_overlap
use_window = getattr(args, 'use_window', False)
batch_size = args.batch_size
num_worker = 4

if use_wandb:
    import wandb

if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(f"Using GPU device(s): {args.gpu}")
else:
    print("Using all available GPUs (CUDA_VISIBLE_DEVICES not set)")

torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
today = datetime.datetime.today().strftime('%Y-%m-%d')

print("task_name:", task_name)
print("experiment:", experiment)
print("Input_mode:", input_mode, "train:", train, "val:", val)

if dataset == 1:
    dataset_name = "dataset1"
    args.data_path = "../dataset/dataset1/"
    args.model_path = f"./saved_models/dataset1/{experiment}/"
    args.log_path = f"./runs/dataset1/{experiment}/"
elif dataset == 2:
    dataset_name = "dataset2"
    args.data_path = "../dataset/dataset2/"
    args.model_path = f"./saved_models/dataset2/{experiment}/"
    args.log_path = f"./runs/dataset2/{experiment}/"

data_path = args.data_path
model_path = args.model_path
log_path = args.log_path
model_paths = {'pre': os.path.join(model_path, args.pre_path or "")}

os.makedirs(model_path, exist_ok=True)
os.makedirs(log_path, exist_ok=True)
print("\nn_class:", n_class)
print("data_path:", data_path)
print("model_path:", model_path)
print("log_path:", log_path, "\n")

if train and use_wandb:
    wandb.init(
        project="SAMHA",
        name=f"{dataset_name}_{experiment}_{task_name}",
        config=vars(args),
    )
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("train-epoch/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")
print("preparing datasets and dataloaders......")

all_ids = [f for f in os.listdir(os.path.join(data_path, "train", "images")) if is_image_file(f)]
ids_train, ids_val = train_test_split(all_ids, test_size=0.2, random_state=42)
ids_test = [f for f in os.listdir(os.path.join(data_path, "test", "images")) if is_image_file(f)]
print(f"Dataset splits - Train: {len(ids_train)}, Val: {len(ids_val)}, Test: {len(ids_test)}")

dataset_train = DATALOADER(dataset, os.path.join(data_path, "train"), ids=ids_train, label=True)
dataset_val = DATALOADER(dataset, os.path.join(data_path, "train"), ids=ids_val, label=True)
dataset_test = DATALOADER(dataset, os.path.join(data_path, "test"), ids=ids_test, label=True)

if dataset == 2:
    dataset_train.wsi_level = args.wsi_level
    dataset_val.wsi_level = args.wsi_level
    dataset_test.wsi_level = args.wsi_level

persistent = num_worker > 0
dataloader_train = torch.utils.data.DataLoader(
    dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate,
    num_workers=num_worker, pin_memory=True, persistent_workers=persistent
)
dataloader_val = torch.utils.data.DataLoader(
    dataset_val, batch_size=batch_size, shuffle=False, collate_fn=collate,
    num_workers=num_worker, pin_memory=True, persistent_workers=persistent
)
dataloader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, collate_fn=collate,
    num_workers=num_worker, pin_memory=True, persistent_workers=persistent
)

print(f"\nDataloader lengths:")
print(f"  Train: {len(dataloader_train)} batches ({len(dataset_train)} images, batch_size={batch_size})")
print(f"  Val: {len(dataloader_val)} batches ({len(dataset_val)} images, batch_size={batch_size})")
print(f"  Test: {len(dataloader_test)} batches ({len(dataset_test)} images, batch_size=1)")
print()

print("creating models......")
print("Model paths:")
for name, path in model_paths.items():
    print(f"  {name}: {path}")

model = create_model_load_weights(n_class, pre_path=model_paths["pre"], input_mode=input_mode, use_window=use_window)
optimizer, scheduler = get_optimizer_and_scheduler(model=model, base_learning_rate=args.lr, num_epochs=num_epochs, iters_per_epoch=len(dataloader_train))

#class_weights = [0.5127502165862436, 20.107510061415578] #dataset2
class_weights = [0.5594677721318467, 4.703957724290087] #dataset1

print(f"Using class weights: {[f'{w:.6f}' for w in class_weights]}\n\n")

focal_loss = FocalLoss(gamma=3, class_weights=class_weights)

if val:
    f_log = open(log_path + task_name + ".log", 'w')
    f_log.write("Args configuration:\n")
    for key, value in vars(args).items():
        f_log.write(f"  {key}: {value}\n")
    f_log.write("\n")
    f_log.flush()

trainer = Trainer(optimizer, focal_loss, n_class, size_p, size_g, sub_batch_size, input_mode, dataset, context_M, context_L, patch_overlap)
evaluator = Evaluator(n_class, size_p, size_g, sub_batch_size, input_mode, train, dataset, context_M, context_L, patch_overlap)
evaluator_test = Evaluator(n_class, size_p, size_g, sub_batch_size, input_mode, train, dataset, context_M, context_L, patch_overlap)

best_pred = 0.0

interval_start_time = None
interval_start_epoch = None

def format_time(seconds):
    if seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"

for epoch in range(0, num_epochs):
    if not train:
        break
    
    if epoch == 0:
        interval_start_time = time.time()
        interval_start_epoch = 1
    
    if input_mode in (1, 2, 3):
        trainer.set_train(model)
    else:
        raise ValueError(f"Invalid Input_mode: {input_mode}. Must be 1-3.")
    
    optimizer.zero_grad()
    
    tbar = tqdm(dataloader_train)
    
    running_loss = 0.0
    ep_miou_sum = ep_mprec_sum = ep_mrec_sum = ep_mdice_sum = 0.0
    ep_batches = 0
    
    for i_batch, sample_batched in enumerate(tbar):
        loss = trainer.train(sample_batched, model)
        
        if isinstance(scheduler, dict):
            for opt_name, sched in scheduler.items():
                sched(optimizer[opt_name], i_batch, epoch, best_pred)
        else:
            scheduler(optimizer, i_batch, epoch, best_pred)
        
        batch_loss = loss.item() if hasattr(loss, 'item') else float(loss)
        running_loss += batch_loss
        
        s = trainer.get_scores()
        miou = float(s["miou_excl_bg"])
        mprec = float(s["mprec_excl_bg"])
        mrec = float(s["mrec_excl_bg"])
        mdice = float(s["mdice_excl_bg"])
    
        tbar.set_description(
            f"Epoch {epoch} | Loss: {running_loss / (i_batch + 1):.4f} | "
            f"mIoU: {miou:.3f} | mPrec: {mprec:.3f} | mRec: {mrec:.3f} | mDice: {mdice:.3f}"
        )
        
        if use_wandb:
            current_lr = float(optimizer.param_groups[0]['lr'])
            wandb.log({
                "epoch": epoch, "train/loss": batch_loss, "train/lr": current_lr,
                "train/miou": miou, "train/precision": mprec, "train/recall": mrec, "train/dice": mdice,
            })
        
        ep_miou_sum += miou
        ep_mprec_sum += mprec
        ep_mrec_sum += mrec
        ep_mdice_sum += mdice
        ep_batches += 1

    s = trainer.get_scores()
    trainer.reset_metrics()
    
    if ep_batches > 0:
        ep_miou = ep_miou_sum / ep_batches
        ep_mprec = ep_mprec_sum / ep_batches
        ep_mrec = ep_mrec_sum / ep_batches
        ep_mdice = ep_mdice_sum / ep_batches
        ep_loss = running_loss / ep_batches
    else:
        ep_miou = ep_mprec = ep_mrec = ep_mdice = ep_loss = 0.0
    
    if use_wandb:
        wandb.log({
            "epoch": epoch, "train-epoch/miou": ep_miou, "train-epoch/precision": ep_mprec,
            "train-epoch/recall": ep_mrec, "train-epoch/dice": ep_mdice, "train-epoch/loss": ep_loss,
        })
    
    if (epoch + 1) % 5 == 0:
        current_epoch = epoch + 1
        
        if interval_start_time is not None:
            interval_end_time = time.time()
            total_interval_time = interval_end_time - interval_start_time
        else:
            interval_start_time = time.time()
            interval_start_epoch = 1
            total_interval_time = 0
        
        with torch.no_grad():
            print(f"\nValidating at epoch {current_epoch}...")
            
        if input_mode in (1, 2, 3):
            model.eval()
        else:
            raise ValueError(f"Invalid Input_mode: {input_mode}. Must be 1-3.")
            
        evaluator.reset_metrics()
        tbar = tqdm(dataloader_val)
        sample_logged = False
        
        for i_batch, sample_batched in enumerate(tbar):
            predictions = evaluator.eval_test(sample_batched, model)
            
            score_val = evaluator.get_scores()
            mean_iou_VAL = float(score_val["miou_excl_bg"])
            mean_prec_VAL = float(score_val["mprec_excl_bg"])
            mean_rec_VAL = float(score_val["mrec_excl_bg"])
            mean_dice_VAL = float(score_val["mdice_excl_bg"])
            
            tbar.set_description('mIoU: %.3f' % (mean_iou_VAL))

        if mean_iou_VAL > best_pred:
            best_pred = mean_iou_VAL
            ckpt_path = os.path.join(model_path, f"{task_name}.pth")
            
            if input_mode in (1, 2, 3):
                torch.save(model.state_dict(), ckpt_path)
            else:
                raise ValueError(f"Invalid Input_mode: {input_mode}. Must be 1-3.")
        
        score_val_full = evaluator.get_scores()
        evaluator.reset_metrics()

        if use_wandb:
            wandb.log({
                "epoch": epoch, "val/miou": mean_iou_VAL, "val/precision": mean_prec_VAL,
                "val/recall": mean_rec_VAL, "val/dice": mean_dice_VAL, "val/best_miou": best_pred,
            })
        
        log = (
            f'epoch [{epoch+1}/{num_epochs}] time={format_time(total_interval_time)} '
            f'train mIoU={ep_miou:.4f}   mP={ep_mprec:.4f}   mR={ep_mrec:.4f}   mDice={ep_mdice:.4f} | '
            f'val mIoU={mean_iou_VAL:.4f}   mP={mean_prec_VAL:.4f}   mR={mean_rec_VAL:.4f}   mDice={mean_dice_VAL:.4f}\n'
        )
        print(log)
        if val:
            f_log.write(log)
            f_log.flush()
        
        interval_start_time = time.time()
        interval_start_epoch = current_epoch + 1

if val:
    f_log.close()

if not train:
    with torch.no_grad():
        print("testing...")
        if input_mode in (1, 2, 3):
            model.eval()
        else:
           raise ValueError(f"Invalid Input_mode: {input_mode}. Must be 1-3.")

        evaluator_test.reset_metrics()
        tbar = tqdm(dataloader_test)
        print(f"len testloader: {len(dataloader_test)}")
        
        pred_root = "./prediction"
        pred_save_dir = os.path.join(pred_root, dataset_name, experiment, task_name)
        os.makedirs(pred_save_dir, exist_ok=True)
        
        total_miou_excl_bg = total_mprec_excl_bg = total_mrec_excl_bg = total_mdice_excl_bg = 0.0
        total_miou_incl_bg = total_mprec_incl_bg = total_mrec_incl_bg = total_mdice_incl_bg = 0.0
        total_images = 0
        
        fmt = lambda arr: "[" + ", ".join(f"{v:.4f}" for v in np.nan_to_num(np.atleast_1d(arr).astype(float))) + "]"
        
        test_log_path = os.path.join(pred_save_dir, "test_results.txt")
        with open(test_log_path, "w") as f_test:
            f_test.write("Test Results (bg excluded and included)\n")
            f_test.write("Image ID | IoU_per_C | Precision_per_C | Recall_per_C | Dice_per_C | mIoU_noBG | mIoU_wBG | mPrec_noBG | mRec_noBG | mDice_noBG | mPrec_wBG | mRec_wBG | mDice_wBG\n")
            
            for sample in tbar:
                stored_img_name = sample.get("img_name", None)
                if isinstance(stored_img_name, (list, tuple)):
                    stored_img_name = stored_img_name[0]
                
                raw_ids = sample.get("id", ["unknown_id"])
                if isinstance(raw_ids, (list, tuple)):
                    raw_id = raw_ids[0]
                else:
                    raw_id = raw_ids
                    sample["id"] = [raw_id]

                if "image" in sample and not isinstance(sample["image"], (list, tuple)):
                    sample["image"] = [sample["image"]]
                if "label" in sample and not isinstance(sample["label"], (list, tuple)):
                    sample["label"] = [sample["label"]]

                image_obj = sample["image"][0] if sample["image"] else None
                label_obj = sample["label"][0] if sample["label"] else None

                image_path = getattr(image_obj, "filename", None)
                label_path = getattr(label_obj, "filename", None)

                if stored_img_name:
                    pred_filename = stored_img_name
                elif label_path:
                    pred_filename = os.path.basename(label_path)
                elif image_path:
                    pred_filename = os.path.basename(image_path)
                else:
                    pred_filename = f"{raw_id}.png"
                
                base_name, ext = os.path.splitext(pred_filename)
                pred_filename = base_name + (ext if ext else ".png")

                preds = evaluator_test.eval_test(sample, model)
                scores = evaluator_test.get_scores()
                
                miou_excl_bg = float(scores["miou_excl_bg"])
                mprec_excl_bg = float(scores["mprec_excl_bg"])
                mrec_excl_bg = float(scores["mrec_excl_bg"])
                mdice_excl_bg = float(scores["mdice_excl_bg"])
                
                miou_incl_bg = float(scores["miou_incl_bg"])
                mprec_incl_bg = float(scores["mprec_incl_bg"])
                mrec_incl_bg = float(scores["mrec_incl_bg"])
                mdice_incl_bg = float(scores["mdice_incl_bg"])
                
                pred_np = preds[0] if isinstance(preds, (list, tuple)) else preds
                if torch.is_tensor(pred_np):
                    pred_np = pred_np.detach().cpu().numpy()

                if n_class == 2:
                    pred_mask = (pred_np > 0).astype(np.uint8) * 255
                    pred_save_path = os.path.join(pred_save_dir, pred_filename)
                    Image.fromarray(pred_mask, mode='L').save(pred_save_path)
                else:
                    pred_vis = classToRGB(dataset, pred_np)
                    pred_save_path = os.path.join(pred_save_dir, pred_filename)
                    transforms.functional.to_pil_image(pred_vis).save(pred_save_path)
                
                iou_c = scores["iou_per_class"]
                prec_c = scores["precision_per_class"]
                rec_c = scores["recall_per_class"]
                dice_c = scores["dice_per_class"]
                
                f_test.write(
                    f"{base_name} | {fmt(iou_c)} | {fmt(prec_c)} | {fmt(rec_c)} | {fmt(dice_c)} | "
                    f"{miou_excl_bg:.4f} | {miou_incl_bg:.4f} | {mprec_excl_bg:.4f} | {mrec_excl_bg:.4f} | {mdice_excl_bg:.4f} | "
                    f"{mprec_incl_bg:.4f} | {mrec_incl_bg:.4f} | {mdice_incl_bg:.4f}\n"
                )
                
                total_miou_excl_bg += miou_excl_bg
                total_mprec_excl_bg += mprec_excl_bg
                total_mrec_excl_bg += mrec_excl_bg
                total_mdice_excl_bg += mdice_excl_bg
                
                total_miou_incl_bg += miou_incl_bg
                total_mprec_incl_bg += mprec_incl_bg
                total_mrec_incl_bg += mrec_incl_bg
                total_mdice_incl_bg += mdice_incl_bg
                
                total_images += 1
                evaluator_test.reset_metrics()
            
            f_test.write("\n---- Summary over all images ----\n\n")
            if total_images > 0:
                f_test.write(f"Overall Mean IoU (no-bg): {total_miou_excl_bg/total_images:.4f}\n")
                f_test.write(f"Overall Mean Precision (no-bg): {total_mprec_excl_bg/total_images:.4f}\n")
                f_test.write(f"Overall Mean Recall (no-bg): {total_mrec_excl_bg/total_images:.4f}\n")
                f_test.write(f"Overall Mean Dice (no-bg): {total_mdice_excl_bg/total_images:.4f}\n")
                f_test.write(f"\nOverall Mean IoU (with-bg): {total_miou_incl_bg/total_images:.4f}\n")
                f_test.write(f"Overall Mean Precision (with-bg): {total_mprec_incl_bg/total_images:.4f}\n")
                f_test.write(f"Overall Mean Recall (with-bg): {total_mrec_incl_bg/total_images:.4f}\n")
                f_test.write(f"Overall Mean Dice (with-bg): {total_mdice_incl_bg/total_images:.4f}\n")
            else:
                f_test.write("No images processed.\n")

if train and use_wandb:
    wandb.summary["best_miou"] = best_pred
    wandb.finish()