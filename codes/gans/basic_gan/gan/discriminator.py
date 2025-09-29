import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.discriminator(x)
    