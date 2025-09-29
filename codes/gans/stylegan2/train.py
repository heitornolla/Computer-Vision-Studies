#https://blog.paperspace.com/implementation-stylegan2-from-scratch/

import torch
import src.utils as utils

from torch import optim
from tqdm import tqdm

from src.gan.generator import Generator
from src.gan.discriminator import Discriminator
from src.gan.modules.mapping_network import MappingNetwork
from src.gan.modules.path_length_penalty import PathLengthPenalty

DATASET                 = ""
DEVICE                  = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS                  = 500
LEARNING_RATE           = 1e-3
BATCH_SIZE              = 16 # experiment other values
LOG_RESOLUTION          = 7 # for 128*128, other values for bigger imgs
Z_DIM                   = 256 # original was 512
W_DIM                   = 256 # original was 512
LAMBDA_GP               = 10


def train_fn(
    critic,
    gen,
    path_length_penalty,
    loader,
    opt_critic,
    opt_gen,
    opt_mapping_network,
):
    loop = tqdm(loader, leave=True)

    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(DEVICE)
        cur_batch_size = real.shape[0]

        w     = utils.get_w(cur_batch_size, mapping_network)
        noise = utils.get_noise(cur_batch_size)
        with torch.amp.autocast(device_type='cuda'):
            fake = gen(w, noise)
            critic_fake = critic(fake.detach())
            
            critic_real = critic(real)
            gp = utils.gradient_penalty(critic, real, fake, device=DEVICE)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + LAMBDA_GP * gp
                + (0.001 * torch.mean(critic_real ** 2))
            )

        critic.zero_grad()
        loss_critic.backward()
        opt_critic.step()

        gen_fake = critic(fake)
        loss_gen = -torch.mean(gen_fake)

        if batch_idx % 16 == 0:
            plp = path_length_penalty(w, fake)
            if not torch.isnan(plp):
                loss_gen = loss_gen + plp

        mapping_network.zero_grad()
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        opt_mapping_network.step()

        loop.set_postfix(
            gp=gp.item(),
            loss_critic=loss_critic.item(),
        )



if __name__ == "__main__":
    loader              = utils.get_loader(DATASET)

    gen                 = Generator(LOG_RESOLUTION, W_DIM).to(DEVICE)
    critic              = Discriminator(LOG_RESOLUTION).to(DEVICE)
    mapping_network     = MappingNetwork(Z_DIM, W_DIM).to(DEVICE)
    path_length_penalty = PathLengthPenalty(0.99).to(DEVICE)

    opt_gen             = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
    opt_critic          = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
    opt_mapping_network = optim.Adam(mapping_network.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))

    gen.train()
    critic.train()
    mapping_network.train()

    for epoch in range(EPOCHS):
        train_fn(
            critic,
            gen,
            path_length_penalty,
            loader,
            opt_critic,
            opt_gen,
            opt_mapping_network,
        )
        if epoch % 50 == 0:
            utils.generate_examples(gen, epoch, mapping_network=mapping_network)
