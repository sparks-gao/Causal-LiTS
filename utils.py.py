import os
import logging
import datetime
import numpy as np
import torch
import torch.nn.functional as F
import scipy.ndimage as ndi
from scipy.ndimage import label, median_filter

def setup_logger(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = logging.getLogger("CausalSeg")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(ch)
    return logger

def get_patch_instances(patch_mask, grid_size=14):
    mask_2d = patch_mask.view(grid_size, grid_size).cpu().numpy().astype(np.int32)
    labeled, num = ndi.label(mask_2d)
    labels = torch.from_numpy(labeled.reshape(-1)).to(patch_mask.device)
    return labels, num

def get_instances(mask):
    labeled, num = label(mask.astype(np.int32))
    return labeled, num

def dice_binary(pred, gt, eps=1e-6):
    inter = (pred & gt).sum()
    return (2.0 * inter + eps) / (pred.sum() + gt.sum() + eps)

def iou_binary(pred, gt, eps=1e-6):
    inter = (pred & gt).sum()
    union = pred.sum() + gt.sum() - inter
    return (inter + eps) / (union + eps)

def instance_aware_dice(pred_bin, gt_bin):
    gt_lab, gt_n = get_instances(gt_bin)
    pred_lab, pred_n = get_instances(pred_bin)
    if gt_n == 0:
        return 1.0 if pred_n == 0 else 0.0
    dices = []
    for i in range(1, gt_n + 1):
        gt_i = (gt_lab == i)
        best_dice = 0.0
        for j in range(1, pred_n + 1):
            pred_j = (pred_lab == j)
            d = dice_binary(pred_j, gt_i)
            if d > best_dice:
                best_dice = d
        dices.append(best_dice)
    return float(np.mean(dices))

def instance_aware_iou(pred_bin, gt_bin):
    gt_lab, gt_n = get_instances(gt_bin)
    pred_lab, pred_n = get_instances(pred_bin)
    if gt_n == 0:
        return 1.0 if pred_n == 0 else 0.0
    ious = []
    for i in range(1, gt_n + 1):
        gt_i = (gt_lab == i)
        best_iou = 0.0
        for j in range(1, pred_n + 1):
            pred_j = (pred_lab == j)
            iou = iou_binary(pred_j, gt_i)
            if iou > best_iou:
                best_iou = iou
        ious.append(best_iou)
    return float(np.mean(ious))

@torch.no_grad()
def batch_instance_metrics(pred, gt, threshold=0.3):
    pred_bin = (torch.sigmoid(pred) > threshold).cpu().numpy()
    gt_bin = gt.cpu().numpy()
    dice_list, iou_list = [], []
    B = pred.shape[0]
    for b in range(B):
        d = instance_aware_dice(pred_bin[b,0], gt_bin[b,0])
        i = instance_aware_iou(pred_bin[b,0], gt_bin[b,0])
        dice_list.append(d)
        iou_list.append(i)
    return float(np.mean(dice_list)), float(np.mean(iou_list))

def dice_loss(pred, gt):
    pred = torch.sigmoid(pred)
    inter = (pred * gt).sum()
    return 1 - (2 * inter + 1e-6) / (pred.sum() + gt.sum() + 1e-6)

def dice_score(pred, gt, eps=1e-6):
    pred = (torch.sigmoid(pred) > 0.5).float()
    inter = (pred * gt).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + gt.sum(dim=(1,2,3))
    return ((2 * inter + eps) / (union + eps)).mean().item()

def iou_score(pred, gt, eps=1e-6):
    pred = (torch.sigmoid(pred) > 0.5).float()
    inter = (pred * gt).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + gt.sum(dim=(1,2,3)) - inter
    return ((inter + eps) / (union + eps)).mean().item()

def soft_dice_score(pred, gt, eps=1e-6):
    prob = torch.sigmoid(pred)
    inter = (prob * gt).sum(dim=(1,2,3))
    union = prob.sum(dim=(1,2,3)) + gt.sum(dim=(1,2,3))
    return ((2 * inter + eps) / (union + eps)).mean().item()

def postprocess_3d(pred_vol, min_volume=100, k_largest=1):
    pred_vol = median_filter(pred_vol, size=(3, 1, 1)) 
    labeled, num = label(pred_vol)
    if num == 0:
        return pred_vol
    sizes = np.bincount(labeled.ravel())[1:]
    largest_idx = np.argsort(sizes)[-k_largest:] + 1
    mask = np.isin(labeled, largest_idx)
    pred_vol = pred_vol * mask
    labeled, num = label(pred_vol)
    for i in range(1, num + 1):
        comp = (labeled == i)
        if comp.sum() < min_volume:
            pred_vol[comp] = 0
    return pred_vol