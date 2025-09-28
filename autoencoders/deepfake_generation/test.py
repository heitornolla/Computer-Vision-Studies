import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os

from deepfake_autoencoder import DeepFakeAutoencoder
from utils.dataset import DeepfakeDataset


def plot_reconstructions(orig, recon, output_dir, batch_idx, num_images):
    """
    Given the original and reconstructed images, create and save a plot.
    """

    fig, axes = plt.subplots(2, num_images, figsize=(2 * num_images, 4))

    for j in range(num_images):
        axes[0, j].imshow(orig[j, 0], cmap='gray')
        axes[0, j].axis('off')
        axes[0, j].set_title('Orig')

        axes[1, j].imshow(recon[j, 0], cmap='gray')
        axes[1, j].axis('off')
        axes[1, j].set_title('Recon')

    batch_save_path = os.path.join(output_dir, f"reconstruction_batch_{batch_idx:03d}.png")
    plt.tight_layout()
    plt.savefig(batch_save_path)
    plt.close()
    print(f"Saved {batch_save_path}")


def run_inference(input_dir, output_dir, weights="deepfake_ae_best.pth", batch_size=16):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DeepFakeAutoencoder().to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    ds = DeepfakeDataset(input_dir, size=256, b=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for i, imgs in enumerate(loader):
            imgs = imgs.to(device)

            z = model.encoder(imgs)
            out = model.decoder_a(z)

            ### Uncomment to save individual images
            # for j in range(out.size(0)):
            #     out_path = os.path.join(output_dir, f"{i*batch_size + j:05d}.png")
            #     save_image(out[j], out_path)
            #     print(f"Saved {out_path}")

            ### Saves original and reconstructions
            orig = imgs.cpu().numpy()
            recon = out.cpu().numpy()

            num_images = orig.shape[0]

            plot_reconstructions(orig, recon, output_dir, i, num_images)


if __name__ == "__main__":
    input_folder = ""
    output_folder = "deepfake_outputs"
    run_inference(input_folder, output_folder, weights="deepfake_ae_best.pth")
