import torch
from torch.utils.data import Dataset
from glob import glob
import cv2
import os


class DeepfakeDataset(Dataset):
    def __init__(self, folder, size=256, b=False, label=None):
        self.size = size
        self.label = 'a' if b is False else 'b'
        if b:
            self.fpaths = glob(os.path.join(folder, "*", "*.png"))
        else:
            self.fpaths = glob(os.path.join(folder, "*.png"))
        self.fpaths = [f for f in self.fpaths if 'slap' not in os.path.basename(f)]

        valid_fpaths = []
        for f in self.fpaths:
            im = cv2.imread(f)
            if im is None:
                continue
            h, w = im.shape[:2]
            if h >= size and w >= size:
                valid_fpaths.append(f)
        self.fpaths = valid_fpaths

        print(f"[{self.label}] Found {len(self.fpaths)} images.")

    def center_crop(self, im):
        h, w = im.shape[:2]
        start_y = (h - self.size) // 2
        start_x = (w - self.size) // 2
        return im[start_y:start_y+self.size, start_x:start_x+self.size]

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, idx):
        f = self.fpaths[idx]
        im = cv2.imread(f)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = self.center_crop(im)
        im = torch.tensor(im / 255.0, dtype=torch.float32).permute(2, 0, 1)
        return im
