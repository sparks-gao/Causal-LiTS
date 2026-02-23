import torch

class CFG:
    img_size = 224
    patch_size = 16
    grid_size = 14
    batch_size = 64
    num_epochs = 100
    lr = 1e-4
    weight_decay = 1e-4
    lambda_seg = 1.0
    lambda_contrast = 0.1
    lambda_causal = 0.3
    lambda_boundary = 0.1
    num_workers = 8
    pin_memory = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_prompt_file = "text_prompt.txt"
    save_dir = ""
    data_root = ""