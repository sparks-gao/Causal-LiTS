import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import scipy.ndimage as ndi
from config import CFG
from utils import get_patch_instances

class CLIPFeatureExtractor(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model
        self.feats = {}
        def hook(name):
            def fn(_, __, out):
                self.feats[name] = out
            return fn
        self.clip.visual.transformer.resblocks[2].register_forward_hook(hook("l3"))
        self.clip.visual.transformer.resblocks[5].register_forward_hook(hook("l6"))

    def forward(self, x):
        _ = self.clip.encode_image(x)
        l3_feat = self.feats["l3"].permute(1, 0, 2)
        l6_feat = self.feats["l6"].permute(1, 0, 2)
        return l3_feat[:, 1:, :].float(), l6_feat[:, 1:, :].float()

@torch.no_grad()
def get_text_direction(clip_model, prompts):
    pos = [p for p in prompts]
    neg = ["a CT image of the liver without tumor"] * len(prompts)
    t_pos = clip.tokenize(pos).to(CFG.device)
    t_neg = clip.tokenize(neg).to(CFG.device)
    f_pos = clip_model.encode_text(t_pos)
    f_neg = clip_model.encode_text(t_neg)
    d = f_pos - f_neg
    d = d / (d.norm(dim=-1, keepdim=True) + 1e-6)
    return d

class CounterfactualModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim//4), nn.ReLU(),
            nn.Linear(dim//4, 1), nn.Sigmoid()
        )

    def forward(self, F, d):
        alpha = self.gate(F)
        proj = torch.einsum("bnc,bc->bn", F, d)
        delta = alpha * proj.unsqueeze(-1) * d.unsqueeze(1)
        return F - delta 

class InstanceAwareContrast(nn.Module):
    def __init__(self, tau=0.07, min_pixels=3, lambda_cf=0.5): 
        super().__init__()
        self.tau = tau
        self.min_pixels = min_pixels
        self.lambda_cf = lambda_cf 

    def forward(self, dp, f_cf, patch_mask): 
        dp = F.normalize(dp, dim=-1)
        f_cf = F.normalize(f_cf, dim=-1) 

        labels, num_inst = get_patch_instances(patch_mask)
        if num_inst == 0:
            return torch.tensor(0., device=dp.device)

        bg_dp = dp[patch_mask == 0]
        bg_cf = f_cf[patch_mask == 0]
        if bg_dp.size(0) == 0:
            return torch.tensor(0., device=dp.device)

        mu_bg_cf = F.normalize(bg_cf.mean(0, keepdim=True), dim=-1) 
        loss_tumor, count_t = 0.0, 0
        loss_cf, count_cf = 0.0, 0

        for k in range(1, num_inst + 1):
            idx = (labels == k)
            if idx.sum() < self.min_pixels:
                continue

            inst_dp = dp[idx]
            inst_cf = f_cf[idx]
            mu_k_dp = F.normalize(inst_dp.mean(0, keepdim=True), dim=-1)
            mu_bg_dp = F.normalize(bg_dp.mean(0, keepdim=True), dim=-1)

            sim_pos_t = inst_dp @ mu_k_dp.t()
            sim_neg_t = inst_dp @ mu_bg_dp.t()
            loss_k_t = -torch.log(
                torch.exp(sim_pos_t / self.tau) /
                (torch.exp(sim_pos_t / self.tau) + torch.exp(sim_neg_t / self.tau))
            ).mean()
            loss_tumor += loss_k_t
            count_t += 1

            mu_k_cf = F.normalize(inst_cf.mean(0, keepdim=True), dim=-1)  
            sim_pos_cf = inst_cf @ mu_bg_cf.t()
            loss_k_cf = -torch.log(  
                torch.exp(sim_pos_cf / self.tau) /
                (torch.exp(sim_pos_cf / self.tau) + torch.exp(inst_cf @ mu_k_cf.t() / self.tau))  
            ).mean()
            loss_cf += loss_k_cf
            count_cf += 1

        if count_t == 0 or count_cf == 0:
            return torch.tensor(0., device=dp.device)
        return loss_tumor / count_t + self.lambda_cf * (loss_cf / count_cf)

class CausalGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.g

def rbf_kernel(x, sigma=None):
    x2 = (x**2).sum(1).view(-1, 1)
    dist = x2 + x2.t() - 2 * torch.matmul(x, x.t())
    dist = torch.clamp(dist, min=0)
    if sigma is None:
        median_dist = torch.median(dist.detach())
        sigma2 = median_dist + 1e-5
    else:
        sigma2 = sigma ** 2
    return torch.exp(-dist / (2 * sigma2))

def hsic(x, y):
    x = F.normalize(x, p=2, dim=1)
    y = F.normalize(y, p=2, dim=1)
    K = rbf_kernel(x)
    L = rbf_kernel(y)
    n = K.size(0)
    H = torch.eye(n, device=x.device) - 1.0 / n
    KH = K @ H
    LH = L @ H
    return torch.sum(KH * LH.t()) / ((n - 1) ** 2)

class InstanceAwareHSICLoss(nn.Module):
    def __init__(self, l1=1e-3, max_bg_samples=32):
        super().__init__()
        self.l1 = l1
        self.max_bg = max_bg_samples

    def forward(self, dp, patch_mask, gate):
        labels, num_inst = get_patch_instances(patch_mask)
        loss = 0.0
        count = 0
        for k in range(1, num_inst + 1):
            idx_inst = (labels == k)
            if idx_inst.sum() < 3:
                continue
            x = dp[idx_inst]
            bg_idx = (patch_mask == 0)
            if bg_idx.sum() < 5:
                continue
            bg = dp[bg_idx]
            if bg.size(0) > self.max_bg:
                bg = bg[torch.randperm(bg.size(0), device=bg.device)[:self.max_bg]]
            if x.size(0) > bg.size(0):
                x = x[torch.randperm(x.size(0), device=x.device)[:bg.size(0)]]
            elif bg.size(0) > x.size(0):
                bg = bg[torch.randperm(bg.size(0), device=bg.device)[:x.size(0)]]
            x = F.normalize(x, dim=1)
            bg = F.normalize(bg, dim=1)
            loss += hsic(x, bg)
            count += 1
        if count == 0:
            return torch.tensor(0., device=dp.device)
        return loss / count + self.l1 * gate.abs().mean()

class Spatial_location_Encoder(nn.Module):
    def __init__(self, in_ch=3, out_ch=256):
        super().__init__()
        self.conv1 = self.block(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = self.block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = self.block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = self.block(256, out_ch)

    def block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU()
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.pool1(x1)
        x2 = self.conv2(x2)
        x3 = self.pool2(x2)
        x3 = self.conv3(x3)
        x4 = self.pool3(x3)
        return self.conv4(x4)

class CausalUNetDecoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.in_conv = nn.Conv2d(dim*4 + 256, 256, 3, padding=1) 
        self.up1 = self.block(256+dim, 128)
        self.up2 = self.block(128+dim, 64)
        self.up3 = self.block(64+dim, 32)
        self.out = nn.Conv2d(32, 1, 1)

    def block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
        )

    def forward(self, F_low, F_mid, F_mid_inst, dp, F_spatial):
        F_spatial = F.interpolate(F_spatial, size=F_low.shape[2:], mode="bilinear")
        x = torch.cat([F_low, F_mid, F_mid_inst, dp, F_spatial], 1)
        x = self.in_conv(x)
        
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        F_low_28 = F.interpolate(F_low, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = self.up1(torch.cat([x, F_low_28], 1))
        
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        F_low_56 = F.interpolate(F_low, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = self.up2(torch.cat([x, F_low_56], 1))
        
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        F_low_112 = F.interpolate(F_low, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = self.up3(torch.cat([x, F_low_112], 1))
        
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.out(x)

def instance_pooling_single(F, patch_mask, grid_size=14, min_pixels=3):
    labels, num_inst = get_patch_instances(patch_mask, grid_size)
    F_out = F.clone()
    for k in range(1, num_inst + 1):
        idx = (labels == k)
        if idx.sum() < min_pixels:
            continue
        proto = F[idx].mean(0, keepdim=True)
        F_out[idx] = proto
    return F_out

def instance_pooling_batch(F, patch_mask):
    B, _, _ = F.shape
    F_inst = torch.zeros_like(F)
    for i in range(B):
        F_inst[i] = instance_pooling_single(F[i], patch_mask[i])
    return F_inst

class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        pred = torch.sigmoid(pred)
        gt_dist = ndi.distance_transform_edt(gt.cpu().numpy())
        pred_dist = ndi.distance_transform_edt((pred > 0.5).cpu().numpy())
        gt_dist = torch.from_numpy(gt_dist).to(pred.device)
        pred_dist = torch.from_numpy(pred_dist).to(pred.device)
        gt_dist = gt_dist / (gt_dist.max() + 1e-6)
        pred_dist = pred_dist / (pred_dist.max() + 1e-6)
        boundary_mask = (gt_dist < 0.1) | (pred_dist < 0.1) 
        loss = torch.abs(gt_dist - pred_dist)[boundary_mask].mean()
        return loss

class CausalCLIPSeg(nn.Module):
    def __init__(self, clip_model, prompt_dict):
        super().__init__()
        self.clip = clip_model
        self.clip_feat = CLIPFeatureExtractor(clip_model).float()
        visual_dim = clip_model.visual.transformer.width
        text_dim = clip_model.text_projection.shape[1]
        self.text_adapter = nn.Linear(text_dim, visual_dim)
        self.cf = CounterfactualModule(visual_dim)
        self.proto = InstanceAwareContrast(tau=0.07)
        self.gate = CausalGate(visual_dim)
        self.causal_loss = InstanceAwareHSICLoss(l1=1e-3)
        self.spatial_encoder = Spatial_location_Encoder(in_ch=3, out_ch=256)
        self.decoder = CausalUNetDecoder(visual_dim)
        self.boundary_loss = BoundaryLoss() 

    def forward(self, x, patch_mask, spur_placeholder, prompts):
        d = get_text_direction(self.clip, prompts).float()
        d = self.text_adapter(d)
        
        F_low, F_mid = self.clip_feat(x)
        B, N, C = F_mid.shape
        F_spatial = self.spatial_encoder(x) 
        
        tumor_ratio = patch_mask.float().mean(dim=1)
        SMALL_THR, LARGE_THR = 0.25, 0.45
        w_contrast = torch.ones_like(tumor_ratio)
        w_causal = torch.ones_like(tumor_ratio)
        w_contrast[tumor_ratio > LARGE_THR] = 0.0
        w_causal[tumor_ratio > LARGE_THR] = 0.0
        mid = (tumor_ratio > SMALL_THR) & (tumor_ratio <= LARGE_THR)
        w_contrast[mid] = (LARGE_THR - tumor_ratio[mid]) / (LARGE_THR - SMALL_THR)
        w_causal[mid] = w_contrast[mid]
        
        F_mid_cf = self.cf(F_mid, d)
        dp = F_mid_cf - F_mid
        F_mid_inst = instance_pooling_batch(F_mid_cf, patch_mask)
        
        loss_contrast = torch.tensor(0., device=x.device)
        for i in range(B):
            if w_contrast[i] > 0:
                loss_contrast += w_contrast[i] * self.proto(dp[i], F_mid_cf[i], patch_mask[i])
                loss_contrast /= (w_contrast.sum() + 1e-6)
                
        loss_causal = torch.tensor(0., device=x.device)
        valid = 0
        for i in range(B):
            if w_causal[i] == 0:
                continue
            loss_i = self.causal_loss(self.gate(dp[i]), patch_mask[i], self.gate.g)
            loss_causal += w_causal[i] * loss_i
            valid += 1
        loss_causal /= max(valid, 1)
        
        def to_map(f):
            return f.transpose(1, 2).reshape(B, C, CFG.grid_size, CFG.grid_size)
            
        mask_pred = self.decoder(
            to_map(F_low),
            to_map(F_mid),
            to_map(F_mid_inst),
            to_map(self.gate(dp).view(B, N, C)),
            F_spatial 
        )
        return mask_pred, loss_contrast, loss_causal
