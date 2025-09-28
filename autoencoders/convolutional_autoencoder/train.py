import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader

import dataset 
from cae import ConvAutoencoder

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_path = ''

data = dataset.AutoencoderDataset(data_path)

data_loader = DataLoader(data,
                        batch_size=32,
                        drop_last=True,
                        shuffle=True)

dataiter = iter(data_loader)
images = next(dataiter)


def get_model():
    model = ConvAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    return model, criterion, optimizer


val_ratio = 0.2
n_total = len(data)
n_val = int(n_total * val_ratio)
n_train = n_total - n_val
train_data, val_data = random_split(data, [n_train, n_val])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, drop_last=True, num_workers=4)
val_loader   = DataLoader(val_data, batch_size=32, shuffle=False, drop_last=True, num_workers=4)

model, criterion, optimizer = get_model()

num_epochs = 50
best_val_loss = float("inf")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for imgs in train_loader:
        imgs = imgs.to(device)
        recon = model(imgs)
        loss = criterion(recon, imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs in val_loader:
            imgs = imgs.to(device)
            recon = model(imgs)
            loss = criterion(recon, imgs)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f"cae_best.pth")
        print(f"Saved new best model at epoch {epoch+1} (val_loss={val_loss:.4f})")
