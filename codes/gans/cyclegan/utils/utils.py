# utils.py — updated for PyTorch 2.5.1
# Notes:
# - Removes deprecated `.data` usage
# - Uses `torch.inference_mode()` for sampling (no grad)
# - Uses robust device discovery and non-blocking transfers
# - Replaces shell mkdir with `os.makedirs(..., exist_ok=True)`
# - Keeps file I/O formats identical to your original code

import os
import warnings

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

warnings.filterwarnings("ignore")


def checkpoint(checkpoint_dir, epoch, G_XtoY, G_YtoX, Dp_X, Dp_Y, Dg_X, Dg_Y, best: bool = False):
    """Save parameters of generators (G_XtoY, G_YtoX) and discriminators (Dp_X, Dp_Y, Dg_X, Dg_Y).

    Creates a subdir named by zero-padded epoch (e.g., '000123') or 'best/'.
    """
    checkpoint_dir = os.path.join(checkpoint_dir, "best" if best else str(epoch).zfill(6))
    os.makedirs(checkpoint_dir, exist_ok=True)

    torch.save(G_XtoY.state_dict(), os.path.join(checkpoint_dir, "G_XtoY.pkl"))
    torch.save(G_YtoX.state_dict(), os.path.join(checkpoint_dir, "G_YtoX.pkl"))
    torch.save(Dp_X.state_dict(), os.path.join(checkpoint_dir, "Dp_X.pkl"))
    torch.save(Dp_Y.state_dict(), os.path.join(checkpoint_dir, "Dp_Y.pkl"))
    torch.save(Dg_X.state_dict(), os.path.join(checkpoint_dir, "Dg_X.pkl"))
    torch.save(Dg_Y.state_dict(), os.path.join(checkpoint_dir, "Dg_Y.pkl"))


@torch.no_grad()
def _tensor_to_uint8_numpy(x: torch.Tensor) -> np.ndarray:
    """Convert a torch.Tensor image/batch in approximately [-0.5, 0.5] to uint8 [0, 255] NCHW.

    - Detaches and moves to CPU
    - Applies ((x + 0.5) * 255), with clamping for safety
    - Returns numpy uint8 with same shape as input
    """
    if x.is_floating_point():
        x = (x.detach().cpu() + 0.5).mul(255.0).clamp(0, 255)
    else:
        x = x.detach().cpu()
    return x.to(dtype=torch.uint8).numpy()


def save_samples(samples_dir, epoch: int, fixed_Y: torch.Tensor, fixed_X: torch.Tensor,
                 G_YtoX: torch.nn.Module, G_XtoY: torch.nn.Module, batch_size: int = 16) -> None:
    """Save sample grids: X->Y->X and Y->X->Y.

    - Uses the *model's* device automatically (no device mismatch).
    - Disables grad via `torch.inference_mode()` for speed/memory.
    - Keeps your matplotlib + OpenCV concatenation pipeline.
    """
    os.makedirs(samples_dir, exist_ok=True)

    # Infer device from model to avoid mismatches
    try:
        device = next(G_XtoY.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Forward passes without grad for sampling
    with torch.inference_mode():
        x_in = fixed_X.to(device, non_blocking=True)
        y_in = fixed_Y.to(device, non_blocking=True)

        fake_Y = G_XtoY(x_in)
        recon_Y_X = G_YtoX(fake_Y)

        fake_X = G_YtoX(y_in)
        recon_X_Y = G_XtoY(fake_X)

    # Convert tensors to numpy uint8
    X = _tensor_to_uint8_numpy(fixed_X)
    Y = _tensor_to_uint8_numpy(fixed_Y)
    fake_Y_np = _tensor_to_uint8_numpy(fake_Y)
    recon_Y_X_np = _tensor_to_uint8_numpy(recon_Y_X)
    fake_X_np = _tensor_to_uint8_numpy(fake_X)
    recon_X_Y_np = _tensor_to_uint8_numpy(recon_X_Y)

    n_rows = min(4, batch_size)
    plt.figure(figsize=(16, 8))
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    for i in range(min(n_rows, batch_size)):
        # X -> Y -> X
        plt.subplot(n_rows * 2, 1, i * 2 + 1)
        plt.title("Original Image X   |   Translated Image    |   Reconstructed Image", fontsize=16, fontweight="bold")
        img_concat = cv2.hconcat([
            np.transpose(X[i, :, :, :], (1, 2, 0)),
            np.transpose(fake_Y_np[i, :, :, :], (1, 2, 0)),
            np.transpose(recon_Y_X_np[i, :, :, :], (1, 2, 0)),
        ])
        plt.imshow(img_concat)

        # Y -> X -> Y
        plt.subplot(n_rows * 2, 1, i * 2 + 2)
        plt.title("Original Image Y   |   Translated Image    |   Reconstructed Image", fontsize=16, fontweight="bold")
        img_concat = cv2.hconcat([
            np.transpose(Y[i, :, :, :], (1, 2, 0)),
            np.transpose(fake_X_np[i, :, :, :], (1, 2, 0)),
            np.transpose(recon_X_Y_np[i, :, :, :], (1, 2, 0)),
        ])
        plt.imshow(img_concat)

    out_path = os.path.join(samples_dir, f"sample-{epoch:06d}.png")
    plt.savefig(out_path)
    plt.close()

