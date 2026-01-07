import random 
import torch 
import os
import numpy as np
from PIL import Image, ImageOps
from utils import config

class PadOrCropPIL:
    def __init__(self, size=256, fill=(255, 255, 255)):
        self.size = size
        self.fill = fill  # bg color

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size

        if w < self.size or h < self.size:
            pad_w = max(0, (self.size - w) // 2)
            pad_h = max(0, (self.size - h) // 2)
            padding = (pad_w, pad_h, self.size - w - pad_w, self.size - h - pad_h)
            img = ImageOps.expand(img, padding, fill=self.fill)

        elif w > self.size or h > self.size:
            left = (w - self.size) // 2
            top = (h - self.size) // 2
            right = left + self.size
            bottom = top + self.size

            img = img.crop((left, top, right, bottom))

        return img


def save_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
