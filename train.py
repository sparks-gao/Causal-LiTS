import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import clip
from tqdm import tqdm

from config import CFG
from utils import setup_logger, dice_loss, dice_score, soft_dice_score, batch_instance_metrics
from dataset import LiverTumorDataset, load_text_prompts
from models import CausalCLIPSeg

@torch.no_grad()
def evaluate(model, loader, device, epoch=0, threshold=0.3):
    model.eval()
    val_pbar = tqdm(loader, desc=f"Val Epoch {epoch+1}", leave=False, dynamic_ncols=True)
    total_loss, total_dice_inst, total_iou_inst, total_soft_dice = 0.0, 0.0, 0.0, 0.0
    n = 0
    for img, mask, pm, spur, prompts in val_pbar:
        img, mask, pm, spur = img.to(device), mask.to(device), pm.to(device), spur.to(device)
        pred, l_con, l_cau = model(img, pm, spur, prompts)
        l_seg = dice_loss(pred, mask) + F.binary_cross_entropy_with_logits(pred, mask)
        l_boundary = model.boundary_loss(pred, mask)
        loss = (CFG.lambda_seg * l_seg + CFG.lambda_contrast * l_con + 
                CFG.lambda_causal * l_cau + CFG.lambda_boundary * l_boundary)
        
        dice_inst, iou_inst = batch_instance_metrics(pred, mask, threshold=threshold)
        total_loss += loss.item()
        total_dice_inst += dice_inst
        total_iou_inst += iou_inst
        total_soft_dice += soft_dice_score(pred, mask)
        n += 1
    return {
        "loss": total_loss / n, "dice_inst": total_dice_inst / n,
        "iou_inst": total_iou_inst / n, "soft_dice": total_soft_dice / n,
    }

def train():
    logger = setup_logger(CFG.save_dir)
    logger.info(f"Configuration: BS={CFG.batch_size}, LR={CFG.lr}, Device={CFG.device}")
    
    clip_model, _ = clip.load("ViT-B/16", device=CFG.device)
    clip_model.float()
    N_TRAIN_BLOCKS = 3
    for p in clip_model.parameters(): p.requires_grad = False
    num_blocks = len(clip_model.visual.transformer.resblocks)
    for i in range(num_blocks - N_TRAIN_BLOCKS, num_blocks):
        for p in clip_model.visual.transformer.resblocks[i].parameters(): p.requires_grad = True
    for p in clip_model.visual.ln_post.parameters(): p.requires_grad = True
    clip_model.train()

    logger.info("Initializing Model...")
    prompt_dict = load_text_prompts(CFG.text_prompt_file)
    model = CausalCLIPSeg(clip_model, prompt_dict).to(CFG.device)
    
    train_set = LiverTumorDataset(CFG.data_root, "train")
    val_set = LiverTumorDataset(CFG.data_root, "val")
    test_set = LiverTumorDataset(CFG.data_root, "test")
    logger.info(f"Datasets: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")
    
    train_loader = DataLoader(train_set, CFG.batch_size, shuffle=True, num_workers=CFG.num_workers, pin_memory=CFG.pin_memory, drop_last=True)
    val_loader = DataLoader(val_set, CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)
    test_loader = DataLoader(test_set, CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)

    clip_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if "clip_feat.clip.visual" in name: clip_params.append(p)
        else: other_params.append(p)
        
    optimizer = torch.optim.AdamW([
        {"params": clip_params, "lr": CFG.lr * 0.1},
        {"params": other_params, "lr": CFG.lr}
    ], weight_decay=CFG.weight_decay)

    best_val_dice = 0.0
    latest_best_path = None
    
    for epoch in range(CFG.num_epochs):
        model.train()
        epoch_loss, epoch_dice_sum, num_batches = 0, 0, 0
        train_pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{CFG.num_epochs}", leave=False, dynamic_ncols=True, mininterval=1.0)
        
        for img, mask, pm, spur, prompts in train_pbar:
            img, mask, pm, spur = img.to(CFG.device), mask.to(CFG.device), pm.to(CFG.device), spur.to(CFG.device)
            pred, l_con, l_cau = model(img, pm, spur, prompts)
            
            l_seg = dice_loss(pred, mask) + F.binary_cross_entropy_with_logits(pred, mask)
            l_boundary = model.boundary_loss(pred, mask)
            loss = CFG.lambda_seg * l_seg + CFG.lambda_contrast * l_con + CFG.lambda_causal * l_cau + CFG.lambda_boundary * l_boundary
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            curr_dice = dice_score(pred, mask)
            epoch_loss += loss.item()
            epoch_dice_sum += curr_dice
            num_batches += 1
            
            train_pbar.set_postfix({"L": f"{loss.item():.3f}", "Dice": f"{curr_dice:.3f}", "Cau": f"{l_cau.item():.3f}", "Bnd": f"{l_boundary.item():.3f}"})
            
        val_metrics = evaluate(model, val_loader, CFG.device, epoch)
        log_msg = (f"[Epoch {epoch+1:03d}] Train Loss: {epoch_loss/num_batches:.4f} | Train Dice: {epoch_dice_sum/num_batches:.4f} || "
                   f"Val Loss: {val_metrics['loss']:.4f} | Val Dice(inst): {val_metrics['dice_inst']:.4f} | "
                   f"Val IoU(inst): {val_metrics['iou_inst']:.4f} | Val SoftDice: {val_metrics['soft_dice']:.4f}")
        logger.info(log_msg)
        
        if val_metrics["dice_inst"] > best_val_dice:
            best_val_dice = val_metrics["dice_inst"]
            save_path = os.path.join(CFG.save_dir, f"best_model_epoch_{epoch+1:03d}.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"✓ Best model updated -> {save_path}")
            latest_best_path = save_path

    logger.info("\nEvaluating on TEST set with best model...")
    if latest_best_path:
        model.load_state_dict(torch.load(latest_best_path))
    test_metrics = evaluate(model, test_loader, CFG.device)
    logger.info(f"[TEST] Loss: {test_metrics['loss']:.4f} | Dice(inst): {test_metrics['dice_inst']:.4f} | IoU(inst): {test_metrics['iou_inst']:.4f}")

if __name__ == "__main__":
    train()