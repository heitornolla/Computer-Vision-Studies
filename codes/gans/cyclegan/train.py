import os
import torch 
import torch.nn as nn
import torch.optim as optim

from utils.utils import save_checkpoint, load_checkpoint

from torchvision.utils import save_image
from tqdm import tqdm

from model.discriminator import Discriminator
from model.generator import Generator

from utils import config
from utils.dataset import get_fingerprint_loaders


def train(
    disc_R, disc_L, gen_L, gen_R, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, epoch
):
    R_reals = 0
    R_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (reference, latent) in enumerate(loop):
        latent = latent.to(config.DEVICE)
        reference = reference.to(config.DEVICE)

        # Train Discriminators H and Z
        with torch.amp.autocast(device_type=config.DEVICE):
            fake_reference = gen_R(latent)
            D_R_real = disc_R(reference)
            D_R_fake = disc_R(fake_reference.detach())
            R_reals += D_R_real.mean().item()
            R_fakes += D_R_fake.mean().item()
            D_R_real_loss = mse(D_R_real, torch.ones_like(D_R_real))
            D_R_fake_loss = mse(D_R_fake, torch.zeros_like(D_R_fake))
            D_R_loss = D_R_real_loss + D_R_fake_loss

            fake_latent = gen_L(reference)
            D_L_real = disc_L(latent)
            D_L_fake = disc_L(fake_latent.detach())
            D_L_real_loss = mse(D_L_real, torch.ones_like(D_L_real))
            D_L_fake_loss = mse(D_L_fake, torch.zeros_like(D_L_fake))
            D_L_loss = D_L_real_loss + D_L_fake_loss

            # put it togethor
            D_loss = (D_R_loss + D_L_loss) / 2

        opt_disc.zero_grad(set_to_none=True)
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.amp.autocast(device_type=config.DEVICE):
            # adversarial loss for both generators
            D_R_fake = disc_R(fake_reference)
            D_L_fake = disc_L(fake_latent)
            loss_G_R = mse(D_R_fake, torch.ones_like(D_R_fake))
            loss_G_L = mse(D_L_fake, torch.ones_like(D_L_fake))

            # cycle loss
            cycle_latent = gen_L(fake_reference)
            cycle_reference = gen_R(fake_latent)
            cycle_latent_loss = l1(latent, cycle_latent)
            cycle_reference_loss = l1(reference, cycle_reference)

            # analyze need for identity loss
            identity_latent = gen_L(latent)
            identity_reference = gen_R(reference)
            identity_latent_loss = l1(latent, identity_latent)
            identity_reference_loss = l1(reference, identity_reference)

            # add all togethor
            G_loss = (
                loss_G_L
                + loss_G_R
                + cycle_latent_loss * config.LAMBDA_CYCLE
                + cycle_reference_loss * config.LAMBDA_CYCLE
                + identity_reference_loss * config.LAMBDA_IDENTITY
                + identity_latent_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_dir = f"saved_images/{epoch}"
            os.makedirs(save_dir, exist_ok=True)

            # latent -> fake_reference
            save_image(
                latent * 0.5 + 0.5,
                f"{save_dir}/latent_source_{idx}.png"
            )
            save_image(
                fake_reference * 0.5 + 0.5,
                f"{save_dir}/latent_to_reference_{idx}.png"
            )

            # reference -> fake_latent
            save_image(
                reference * 0.5 + 0.5,
                f"{save_dir}/reference_source_{idx}.png"
            )
            save_image(
                fake_latent * 0.5 + 0.5,
                f"{save_dir}/reference_to_latent_{idx}.png"
            )

        if idx % 100 == 0:
            loop.set_postfix(R_real=R_reals / (idx + 1), R_fake=R_fakes / (idx + 1))


def main():
    disc_R = Discriminator(in_channels=3).to(config.DEVICE)
    disc_L = Discriminator(in_channels=3).to(config.DEVICE)
    gen_L = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_R = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)

    opt_disc = optim.Adam(
        list(disc_R.parameters()) + list(disc_L.parameters()),
        lr=config.DISC_LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_L.parameters()) + list(gen_R.parameters()),
        lr=config.GEN_LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_REF,
            gen_R,
            opt_gen,
            config.GEN_LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_LAT,
            gen_L,
            opt_gen,
            config.GEN_LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC_REF,
            disc_R,
            opt_disc,
            config.DISC_LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC_LAT,
            disc_L,
            opt_disc,
            config.DISC_LEARNING_RATE,
        )
    
    train_loader, val_loader = get_fingerprint_loaders()

    g_scaler = torch.amp.GradScaler()
    d_scaler = torch.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train(
            disc_R,
            disc_L,
            gen_L,
            gen_R,
            train_loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            d_scaler,
            g_scaler,
            epoch,
        )

        if config.SAVE_MODEL:
            save_checkpoint(gen_R, opt_gen, filename=config.CHECKPOINT_GEN_REF)
            save_checkpoint(gen_L, opt_gen, filename=config.CHECKPOINT_GEN_LAT)
            save_checkpoint(disc_R, opt_disc, filename=config.CHECKPOINT_DISC_REF)
            save_checkpoint(disc_L, opt_disc, filename=config.CHECKPOINT_DISC_LAT)


if __name__ == "__main__":
    main()
