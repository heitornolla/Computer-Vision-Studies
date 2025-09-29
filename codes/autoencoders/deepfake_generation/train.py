import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.dataset import DeepfakeDataset

from deepfake_autoencoder import DeepFakeAutoencoder

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset_reference = DeepfakeDataset('')
dataset_latent = DeepfakeDataset('', reference=False)

train_loader_reference = DataLoader(dataset_reference, batch_size=16, shuffle=True, drop_last=True)
train_loader_latents = DataLoader(dataset_latent, batch_size=16, shuffle=True, drop_last=True)

model = DeepFakeAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

num_epochs = 50
best_loss = float('inf')


for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    # New training loop: separate loss updates for each decoder
    for imgs_ref, imgs_lat in zip(train_loader_reference, train_loader_latents):
        imgs_ref, imgs_lat = imgs_ref.to(device), imgs_lat.to(device)

        optimizer.zero_grad()
        z_ref = model.encoder(imgs_ref)
        out_ref = model.decoder_reference(z_ref)
        loss_ref = criterion(out_ref, imgs_ref)
        loss_ref.backward()
        optimizer.step()

        running_loss += loss_ref.item()

        # update latent decoder + encoder
        optimizer.zero_grad()
        z_lat = model.encoder(imgs_lat)
        out_lat = model.decoder_latent(z_lat)
        loss_lat = criterion(out_lat, imgs_lat)
        loss_lat.backward()
        optimizer.step()

        running_loss += loss_lat.item()

    avg_loss = running_loss / min(len(train_loader_reference), len(train_loader_latents))
    print(f'Epoch {epoch+1}/{num_epochs} || Loss: {avg_loss:.4f}')

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), 'deepfake_ae_best.pth')
        print(f'Saved model from epoch {epoch+1} loss={avg_loss:.4f}')
