import numpy as np
import torch
from utils import dice_binary, iou_binary, postprocess_3d
from models import CausalCLIPSeg
from dataset import LiverTumorDataset
from config import CFG
from torch.utils.data import DataLoader
import clip

def test_with_postprocess(model, test_loader, device, threshold=0.3, min_volume=100, k_largest=1):
    model.eval()
    vol_pred = [] 
    vol_gt = []    
    
    with torch.no_grad():
        for img, mask, _, _, prompts in test_loader:
            img = img.to(device)
            pred, _, _ = model(img, torch.zeros_like(mask).long().to(device), None, prompts) 
            vol_pred.append((torch.sigmoid(pred).cpu().numpy() > threshold))
            vol_gt.append(mask.cpu().numpy())
            
    vol_pred = np.concatenate(vol_pred, axis=0) 
    vol_gt = np.concatenate(vol_gt, axis=0)
    
    vol_pred_bin = vol_pred.squeeze(1) 
    vol_gt_bin = vol_gt.squeeze(1)
    
    vol_pred_post = postprocess_3d(vol_pred_bin, min_volume, k_largest)
    
    dice_3d = dice_binary(vol_pred_post, vol_gt_bin)
    iou_3d = iou_binary(vol_pred_post, vol_gt_bin)
    
    print(f"3D Postprocessed Dice: {dice_3d:.4f}, IoU: {iou_3d:.4f}")
    return dice_3d, iou_3d

if __name__ == "__main__":
    clip_model, _ = clip.load("ViT-B/16", device=CFG.device)
    clip_model.float()
    prompt_dict = {} 
    model = CausalCLIPSeg(clip_model, prompt_dict).to(CFG.device)
    # model.load_state_dict(torch.load('path/to/best_model.pth'))
    
    test_set = LiverTumorDataset(CFG.data_root, "test")
    test_loader = DataLoader(test_set, CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)
    
    test_with_postprocess(model, test_loader, CFG.device)