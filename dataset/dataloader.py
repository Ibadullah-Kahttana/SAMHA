import os
import time
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from PIL import Image, ImageFile
from torchvision.transforms import ToTensor
import cv2

try:
    import openslide
except ImportError:
    print("Warning: openslide library not found. WSI loading (dataset 2) will use standard PIL.")
    openslide = None

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

def is_image_file(filename):
    return any(filename.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".tif", ".svs", ".ndpi"])

def classToRGB(dataset, label):
    l, w = label.shape[0], label.shape[1]
    colmap = np.zeros(shape=(l, w, 3)).astype(np.float32)

    indices = np.where(label == 1)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [255, 255, 255]
    indices = np.where(label == 0)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [0, 0, 0]
        
    transform = ToTensor();
    return transform(colmap)


def class_to_target(inputs, numClass):
    batchSize, l, w = inputs.shape[0], inputs.shape[1], inputs.shape[2]
    target = np.zeros(shape=(batchSize, l, w, numClass), dtype=np.float32)
    for index in range(numClass):
        indices = np.where(inputs == index)
        temp = np.zeros(shape=numClass, dtype=np.float32)
        temp[index] = 1
        target[indices[0].tolist(), indices[1].tolist(), indices[2].tolist(), :] = temp
    return target.transpose(0, 3, 1, 2)


def label_bluring(inputs):
    batchSize, numClass, height, width = inputs.shape
    outputs = np.ones((batchSize, numClass, height, width), dtype=np.float)
    for batchCnt in range(batchSize):
        for index in range(numClass):
            outputs[batchCnt, index, ...] = cv2.GaussianBlur(inputs[batchCnt, index, ...].astype(np.float), (7, 7), 0)
    return outputs


class DATALOADER(data.Dataset):
    def __init__(self, dataset, root, ids, label=False, transform=None):
        super(DATALOADER, self).__init__()
        self.dataset = dataset
        self.root = root
        self.label = label
        self.ids = ids
        self.transform = transform

    def __getitem__(self, index):
        sample = {}
        img_name = self.ids[index]
        image_path = os.path.join(self.root, "images", img_name)
        sample['id'] = os.path.splitext(img_name)[0]
        sample['img_name'] = img_name

        if self.dataset == 2:
            image, sample = self._load_wsi(image_path, img_name, sample)
        else:
            image = Image.open(image_path).convert("RGB")
            image.filename = image_path
        
        sample['image'] = image

        # Load label/mask
        if self.label:
            sample = self._load_mask(img_name, image, sample)
            
        return sample
    
    def _load_wsi(self, image_path, img_name, sample):
        if openslide is None:
            raise RuntimeError("OpenSlide required for WSI loading")
        try:
            slide = openslide.OpenSlide(image_path)
            desired_level = getattr(self, 'wsi_level', 3)
            if slide.level_count <= desired_level:
                desired_level = slide.level_count - 1

            full_w, full_h = slide.level_dimensions[desired_level]
            sample['downsample'] = slide.level_downsamples[desired_level]
            sample['wsi_level'] = desired_level

            image = slide.read_region((0, 0), desired_level, (full_w, full_h)).convert("RGB")
            image.filename = img_name
            slide.close()
        except Exception as e:
            print(f"Error loading WSI {img_name}: {e}, using PIL fallback")
            image = Image.open(image_path).convert("RGB")
            image.filename = img_name
        return image, sample
    
    def _load_mask(self, img_name, image, sample):
        gt_dir = os.path.join(self.root, "gt")
        
        if self.dataset == 2:
            base, ext = os.path.splitext(img_name)
            mask_name = base + "_mask" + ext
        else:
            mask_name = img_name

        mask_path = os.path.join(gt_dir, mask_name)
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        if self.dataset == 2:
            if openslide is None:
                raise RuntimeError("OpenSlide required for Dataset2")
            mask_slide = openslide.OpenSlide(mask_path)
            desired_level = sample.get('wsi_level', 3)
            mask_w, mask_h = mask_slide.level_dimensions[desired_level]
            label = mask_slide.read_region((0, 0), desired_level, (mask_w, mask_h)).convert("L")
            
            image_w, image_h = image.size
            if (mask_w, mask_h) != (image_w, image_h):
                label = label.resize((image_w, image_h), resample=Image.NEAREST)
            mask_slide.close()
        else:
            label = Image.open(mask_path).convert("L")
        
        label.filename = mask_name
        sample["label"] = label
        return sample


    def __len__(self):
        return len(self.ids)
    