import torch
import numpy as np
import cv2
import glob
from model import CycleGAN
from config import *
from utils import _tensor_to_uint8_numpy

# data path
test_path = ''

# create models
G_XtoY, G_YtoX, Dp_X, Dp_Y, Dg_X, Dg_Y = CycleGAN(n_res_blocks=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G_XtoY.to(device)
G_YtoX.to(device)

def translateDomain(img_x: np.ndarray, model: torch.nn.Module) -> np.ndarray:
    # resize and normalize image x
    img_x = cv2.resize(img_x, image_size)
    img_x = (img_x.astype(np.float32) / 255.0) - 0.5
    torch_img_x = np.moveaxis(img_x, -1, 0)[np.newaxis, ...]  # CHW + batch
    img_x_tensor = torch.from_numpy(torch_img_x).to(device, non_blocking=True)

    model.eval()
    with torch.inference_mode():
        torch_img_y = model(img_x_tensor)
    img_y = _tensor_to_uint8_numpy(torch_img_y[0])[...]
    img_y = np.moveaxis(img_y, 0, 2)
    return img_y

if __name__ == '__main__':
    G_XtoY.load_state_dict(torch.load(generator_x_y_weights, map_location=device))
    print('Loaded pretrained weights')

    img_fnames = glob.glob(test_path)

    for fname in img_fnames:
        img_x = cv2.imread(fname)
        img_orig_size = (img_x.shape[1], img_x.shape[0])
        img_x_rgb = cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB)
        img_y = translateDomain(img_x_rgb, G_XtoY)
        img_y_bgr = cv2.cvtColor(img_y, cv2.COLOR_RGB2BGR)
        img_y_bgr = cv2.resize(img_y_bgr, img_orig_size, interpolation=cv2.INTER_LINEAR)
        img_concat = cv2.vconcat([img_x, img_y_bgr])
        cv2.imshow('Image X -> Y', img_concat)
        cv2.waitKey(0)
