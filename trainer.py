from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn.functional as F

from utils.metrics import ConfusionMatrix
from utils.trainer_utils import (
    global_to_patch,
    global_to_context_patches,
    stitch_patch_predictions_to_global,
    images_transform,
    masks_transform,
)

class Trainer(object):
    def __init__(self, optimizer, criterion, n_class, size_p, size_g, sub_batch_size=6, input_mode=1, dataset=1, context_M=2, context_L=3, patch_overlap=0.20):
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics = ConfusionMatrix(n_class)
        self.n_class = n_class
        self.size_p = size_p
        self.size_g = size_g
        self.sub_batch_size = sub_batch_size
        self.input_mode = input_mode
        self.dataset = dataset
        self.context_M = context_M
        self.context_L = context_L
        self.patch_overlap = patch_overlap

    # while random patche generation a batch may have 1 image in a batch, for that we use this function to handle the batch size
    def _handle_batch_size_one(self, local_patches_var, label_local_patches_var=None, medium_patch_var=None, large_patch_var=None):
        actual_batch_size = local_patches_var.size(0)
        need_slice = False
        
        if actual_batch_size == 1:
            need_slice = True
            local_patches_var = torch.cat([local_patches_var, local_patches_var], dim=0)
            if label_local_patches_var is not None:
                label_local_patches_var = torch.cat([label_local_patches_var, label_local_patches_var], dim=0)
            if medium_patch_var is not None:
                medium_patch_var = torch.cat([medium_patch_var, medium_patch_var], dim=0)
            if large_patch_var is not None:
                large_patch_var = torch.cat([large_patch_var, large_patch_var], dim=0)
        
        return local_patches_var, label_local_patches_var, medium_patch_var, large_patch_var, need_slice
        
    def set_train(self, model):
        if hasattr(model, 'module'):
            model.module.train()
        else:
            model.train()
            
    def get_scores(self):
        score = self.metrics.get_scores()
        return score

    def reset_metrics(self):
        self.metrics.reset()

    def train(self, sample, model):
        images, labels = sample['image'], sample['label']
        labels_npy = masks_transform(labels, numpy=True)
        
        local_patches, local_label_patches, coordinates, templates, sizes, ratios = global_to_patch(images, self.size_p, labels=labels, overlap_percentage=self.patch_overlap)
        predicted_patches = [np.zeros((len(coordinates[i]), self.n_class, self.size_p[0], self.size_p[1])) for i in range(len(images))]
        
        if self.input_mode in (2, 3):
            medium_patches = global_to_context_patches(images, self.size_p, coordinates, self.context_M)
        if self.input_mode == 3:
            large_patches = global_to_context_patches(images, self.size_p, coordinates, self.context_L)
        
        total_loss = 0.0
        total_sub_batches = 0
        
        for i in range(len(images)):
            j = 0
            while j < len(coordinates[i]):
                self.optimizer.zero_grad()
                
                batch_local = local_patches[i][j : j+self.sub_batch_size]
                batch_label = local_label_patches[i][j : j+self.sub_batch_size]
                batch_medium = medium_patches[i][j : j+self.sub_batch_size] if self.input_mode in (2, 3) else None
                batch_large = large_patches[i][j : j+self.sub_batch_size] if self.input_mode == 3 else None
                
                local_patches_var = images_transform(batch_local)
                label_local_patches_var = masks_transform(batch_label)

                medium_patch_var = None
                large_patch_var = None
                
                if self.input_mode in (2, 3):
                    medium_patch_var = images_transform(batch_medium)
                if self.input_mode == 3:
                    large_patch_var = images_transform(batch_large)
                
                local_patches_var, label_local_patches_var, medium_patch_var, large_patch_var, need_slice = self._handle_batch_size_one(
                    local_patches_var, label_local_patches_var, medium_patch_var, large_patch_var
                )
                
                if self.input_mode == 1:
                    output_patches = model.forward(local_patches_var)
                elif self.input_mode == 2:
                    output_patches =  model.forward(local_patches_var, medium_patch_var)
                elif self.input_mode == 3:
                    output_patches = model.forward(local_patches_var, medium_patch_var, large_patch_var)
                else:
                    raise ValueError(f"Invalid Input_mode: {self.input_mode}. Must be 1-3.")

                if need_slice:
                    output_patches = output_patches[:1]
                    label_local_patches_var = label_local_patches_var[:1]

                loss = self.criterion(output_patches, label_local_patches_var)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                total_sub_batches += 1

                actual_output_size = output_patches.size(0)
                pred_np = F.interpolate(output_patches, size=self.size_p, mode='nearest').detach().cpu().numpy()
                predicted_patches[i][j:j+actual_output_size] = pred_np
                j += actual_output_size
        
        scores = stitch_patch_predictions_to_global(predicted_patches, self.n_class, sizes, coordinates, self.size_p, templates=templates)
        predictions = [score.argmax(0) for score in scores]

        self.metrics.update(labels_npy, predictions)
        
        avg_loss = total_loss / total_sub_batches if total_sub_batches > 0 else 0.0
        return avg_loss

class Evaluator(object):
    def __init__(self, n_class, size_p, size_g, sub_batch_size=6, input_mode=1, train=True, dataset=1, context_M=2, context_L=3, patch_overlap=0.20):
        self.metrics = ConfusionMatrix(n_class)
        self.n_class = n_class
        self.size_p = size_p
        self.size_g = size_g
        self.sub_batch_size = sub_batch_size
        self.input_mode = input_mode
        self.train = train
        self.dataset = dataset
        self.context_M = context_M
        self.context_L = context_L
        self.patch_overlap = patch_overlap
    
    def _handle_batch_size_one(self, local_patches_var, label_local_patches_var=None, medium_patch_var=None, large_patch_var=None):
        actual_batch_size = local_patches_var.size(0)
        need_slice = False
        
        if actual_batch_size == 1:
            need_slice = True
            local_patches_var = torch.cat([local_patches_var, local_patches_var], dim=0)
            if label_local_patches_var is not None:
                label_local_patches_var = torch.cat([label_local_patches_var, label_local_patches_var], dim=0)
            if medium_patch_var is not None:
                medium_patch_var = torch.cat([medium_patch_var, medium_patch_var], dim=0)
            if large_patch_var is not None:
                large_patch_var = torch.cat([large_patch_var, large_patch_var], dim=0)
        
        return local_patches_var, label_local_patches_var, medium_patch_var, large_patch_var, need_slice
        
    def get_scores(self):
        score = self.metrics.get_scores()
        return score

    def reset_metrics(self):
        self.metrics.reset()

    def eval_test(self, sample, model):
        images = sample['image']
        labels = sample['label']
        labels_npy = masks_transform(labels, numpy=True)
        
        images = [image.copy() for image in images]
        
        local_patches, coordinates, templates, sizes, ratios = global_to_patch(images, self.size_p, overlap_percentage=self.patch_overlap)
        predicted_patches = [np.zeros((len(coordinates[i]), self.n_class, self.size_p[0], self.size_p[1])) for i in range(len(images))]
        
        if self.input_mode in (2, 3):
            medium_patches = global_to_context_patches(images, self.size_p, coordinates, self.context_M)
        if self.input_mode == 3:
            large_patches = global_to_context_patches(images, self.size_p, coordinates, self.context_L)
                
        for i in range(len(images)):
            j = 0
            while j < len(coordinates[i]):
                batch_local = local_patches[i][j:j+self.sub_batch_size]
                batch_medium = medium_patches[i][j:j+self.sub_batch_size] if self.input_mode in (2, 3) else None
                batch_large = large_patches[i][j:j+self.sub_batch_size] if self.input_mode == 3 else None
                
                local_patches_var = images_transform(batch_local)
                medium_patch_var = None
                large_patch_var = None
                        
                if self.input_mode in (2, 3):
                    medium_patch_var = images_transform(batch_medium)
                if self.input_mode == 3:
                    large_patch_var = images_transform(batch_large)
                
                local_patches_var, _, medium_patch_var, large_patch_var, need_slice = self._handle_batch_size_one(
                    local_patches_var, None, medium_patch_var, large_patch_var
                )

                if self.input_mode == 1:
                    output_patches = model.forward(x_local=local_patches_var)
                elif self.input_mode == 2:
                    output_patches =  model.forward(x_local=local_patches_var, x_medium=medium_patch_var)
                elif self.input_mode == 3:
                    output_patches = model.forward(x_local=local_patches_var, x_medium=medium_patch_var, x_large=large_patch_var)
                else:
                    raise ValueError(f"Invalid Input_mode: {self.input_mode}. Must be 1-3.")
                
                if need_slice:
                    output_patches = output_patches[:1]
                
                actual_output_size = output_patches.size()[0]
                predicted_patches[i][j:j+actual_output_size] = (
                    F.interpolate(output_patches, size=self.size_p, mode='nearest').detach().cpu().numpy()
                )
                j += actual_output_size
                    
        scores = stitch_patch_predictions_to_global(predicted_patches, self.n_class, sizes, coordinates, self.size_p, templates=templates)
        predictions = [score.argmax(0) for score in scores]
        self.metrics.update(labels_npy, predictions)

        return predictions
