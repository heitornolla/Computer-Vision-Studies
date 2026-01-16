import os
import torch
from tqdm import tqdm
from torchvision.utils import save_image

from model.generator import Generator
from utils import config
from utils.dataset import get_fingerprint_loaders
from utils.utils import load_checkpoint


def test():
    gen_L = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)

    load_checkpoint(
        config.CHECKPOINT_GEN_LAT, 
        gen_L, 
        optimizer=None, 
        lr=config.GEN_LEARNING_RATE
    )

    gen_L.eval()

    _, val_loader = get_fingerprint_loaders(lat_path='')

    output_dir = ""
    ref_dir = os.path.join(output_dir, 'reference')
    fake_lat_dir = os.path.join(output_dir, 'fake_latent')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(fake_lat_dir, exist_ok=True)

    print(f"Saving images to {output_dir}")

    with torch.no_grad():
        for idx, (reference, _) in enumerate(tqdm(val_loader, mininterval=5.0)):
            reference = reference.to(config.DEVICE)
            
            fake_latent = gen_L(reference)

            for i in range(fake_latent.size(0)):
                img_id = idx * val_loader.batch_size + i
                
                save_image(
                    reference[i] * 0.5 + 0.5, 
                    f"{ref_dir}/reference_{img_id}.png"
                )

                save_image(
                    fake_latent[i] * 0.5 + 0.5, 
                    f"{fake_lat_dir}/generated_latent_{img_id}.png"
                )

    print("Dataset generated")

if __name__ == "__main__":
    test()
