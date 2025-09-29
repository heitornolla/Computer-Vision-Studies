import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import torchvision
from torchvision.transforms import transforms

from torch.utils.tensorboard import SummaryWriter


from gan.discriminator import Discriminator
from gan.generator import Generator


# GANs are very sensitive to hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 1e-4
z_dim = 64  # play with other values
img_dim = 28 * 28 * 1  # 784, MNIST
batch_size = 32
num_epochs = 100


discriminator = Discriminator(img_dim).to(device)
generator = Generator(z_dim, img_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)

transform = transforms.Compose(
    [
        transforms.ToTensor(), 
        transforms.Normalize((0.5), (0.5))
    ]
)

dataset = torchvision.datasets.MNIST(root='./dataset', transform=transform, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

opt_disc = Adam(discriminator.parameters(), lr=lr)
opt_gen = Adam(generator.parameters(), lr=lr)

criterion = nn.BCELoss()

writer_fake = SummaryWriter('runs/GAN_MNIST/fake')
writer_real = SummaryWriter('runs/GAN_MNIST/real')
step = 0


for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        ### Train Discriminator: max log(D(real)) + log(1 - D(G(z)))
        ## log(D(real))
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = generator(noise)
        disc_real = discriminator(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))

        ## log(1 - D(G(z)))
        disc_fake = discriminator(fake.detach()).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        
        lossD = (lossD_real + lossD_fake) / 2

        opt_disc.zero_grad()
        lossD.backward() # keeping fake calculation
        opt_disc.step()

        
        ### Train Generator min log(1 - D(G(z))) == max log(D(G(z)))
        output = discriminator(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        opt_gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = generator(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1
