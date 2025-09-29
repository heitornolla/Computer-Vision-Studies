import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

from dataset import AutoencoderDataset
from cae import ConvAutoencoder


def evaluate_model(weights_path, data_path, output_dir, batch_size=32, num_images=8, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ConvAutoencoder().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    dataset = AutoencoderDataset(data_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    data_iter = iter(data_loader)
    batch_imgs = next(data_iter)[:num_images].to(device)

    with torch.no_grad():
        recon_batch = model(batch_imgs)

    orig = batch_imgs.cpu().numpy()
    recon = recon_batch.cpu().numpy()

    os.makedirs(output_dir, exist_ok=True)

    # Save results
    fig, axes = plt.subplots(2, num_images, figsize=(2*num_images, 4))
    for i in range(num_images):
        axes[0, i].imshow(orig[i, 0], cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title('Orig')

        axes[1, i].imshow(recon[i, 0], cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title('Recon')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "reconstruction_results.png")
    plt.savefig(save_path)
    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    weights = "cae_best.pth"  
    data = ""  # dataset path
    output = "eval_results"

    evaluate_model(weights, data, output)
