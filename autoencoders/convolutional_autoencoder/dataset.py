import torch
from torch.utils.data import Dataset
from glob import glob
import cv2
import os

class AutoencoderDataset(Dataset):
    def __init__(self, folder):
        self.fpaths = glob(os.path.join(folder, '*', '*.png'))

    def center_crop(self, im, size=256):
        h, w = im.shape[:2]
        start_y = max(0, (h - size) // 2)
        start_x = max(0, (w - size) // 2)
        return im[start_y:start_y+size, start_x:start_x+size]

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, idx):
        f = self.fpaths[idx]
        im = cv2.imread(f)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = self.center_crop(im, size=256)
        im = torch.tensor(im / 255.0, dtype=torch.float32)
        im = im.permute(2, 0, 1)  # HWC -> CHW
        return im
