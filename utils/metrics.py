import numpy as np
import math

class ConfusionMatrix(object):

    def __init__(self, n_classes):
        self.n_classes = n_classes
        # axis = 0: target
        # axis = 1: prediction
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        
    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist.astype(np.float64)

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            tmp = self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)
            self.confusion_matrix += tmp

    def get_scores(self):
        hist = self.confusion_matrix

        tp = np.diag(hist)                             # true positives per class
        fp = hist.sum(axis=0) - tp                     # false positives per class
        fn = hist.sum(axis=1) - tp                     # false negatives per class
        # tn = hist.sum() - (tp + fp + fn)             # not needed here

        # IoU = TP / (TP + FP + FN)
        iou_per_class = tp / np.maximum(tp + fp + fn, 1e-12)
        miou_excl_bg = np.mean(np.nan_to_num(iou_per_class[-1]))
        miou_incl_bg = np.nanmean(iou_per_class)

        # Precision = TP / (TP + FP)
        precision_per_class = tp / np.maximum(tp + fp, 1e-12)
        mprec_excl_bg = np.mean(np.nan_to_num(precision_per_class[-1]))
        mprec_incl_bg = np.nanmean(precision_per_class)
        
        # Recall = TP / (TP + FN)
        recall_per_class = tp / np.maximum(tp + fn, 1e-12)
        mrec_excl_bg = np.mean(np.nan_to_num(recall_per_class[-1]))
        mrec_incl_bg = np.nanmean(recall_per_class)

        # Dice/F1 = 2*TP / (2*TP + FP + FN)
        dice_per_class = (2.0 * tp) / np.maximum(2.0 * tp + fp + fn, 1e-12)
        mdice_excl_bg = np.mean(np.nan_to_num(dice_per_class[-1]))
        mdice_incl_bg = np.nanmean(dice_per_class)


        return {
            "iou_per_class": iou_per_class,
            "miou_excl_bg": miou_excl_bg,
            "miou_incl_bg": miou_incl_bg,
            
            "precision_per_class": precision_per_class,
            "mprec_excl_bg": mprec_excl_bg,
            "mprec_incl_bg": mprec_incl_bg,
            
            "recall_per_class": recall_per_class,
            "mrec_excl_bg": mrec_excl_bg,
            "mrec_incl_bg": mrec_incl_bg,
            
            "dice_per_class": dice_per_class, 
            "mdice_excl_bg": mdice_excl_bg,
            "mdice_incl_bg": mdice_incl_bg,
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))