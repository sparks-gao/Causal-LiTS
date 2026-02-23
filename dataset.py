import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import albumentations as A
from config import CFG

def load_text_prompts(prompt_file):
    prompt_dict = {}
    if os.path.exists(prompt_file):
        with open(prompt_file, "r") as f:
            for line in f:
                line = line.strip()
                if ":" in line:
                    cid, txt = line.split(":", 1)
                    prompt_dict[cid] = txt
    return prompt_dict

class LiverTumorDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.split_dir = os.path.join(root_dir, split)
        self.file_list = sorted(glob.glob(os.path.join(self.split_dir, "*.npz")))
        if len(self.file_list) == 0:
            raise ValueError(f"No .npz files found in {self.split_dir}. Please check the data directory.")
        self.split = split
        prompt_path = "" 
        self.prompt_dict = self._load_prompts(prompt_path)
        if self.split == "train":
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=20, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2),
            ])
        else:
            self.transform = None

    def _load_prompts(self, path):
        return load_text_prompts(path)

    def __len__(self):
        return len(self.file_list)

    def make_patch_mask(self, mask):
        t = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
        p = F.adaptive_avg_pool2d(t, (CFG.grid_size, CFG.grid_size))
        p = (p > 0).float()
        return p.view(-1).long()

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        data = np.load(file_path)
        img = data['image'].astype(np.float32)
        mask = data['mask'].astype(np.float32)
        case_id = str(data['case_id'])
        if self.split == "train" and self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()
        if img.ndim == 2:
            img = img.unsqueeze(0)
        img = img.repeat(3, 1, 1)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        patch_mask = self.make_patch_mask(mask.squeeze().numpy())
        spurious = img.mean().view(1)
        prompt = self.prompt_dict.get(case_id, "A CT image of the liver tumor.")
        return img, mask, patch_mask, spurious, prompt