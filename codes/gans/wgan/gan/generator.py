import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim, img_channels, features):
        super(Generator, self).__init__()
        # Input = N x z_dim x 1 x 1
        self.generator = nn.Sequential(
            self._block(z_dim, features*16, 4, 1, 0), # N x f*16 x 4 x 4
            self._block(features*16, features*8, 4, 2, 1), # 8x8
            self._block(features*8, features*4, 4, 2, 1), # 16x16
            self._block(features*4, features*2, 4, 2, 1), # 32x32
            nn.ConvTranspose2d(
                features*2, img_channels, 4, 2, 1
            ),
            nn.Tanh() # [-1, 1]
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.generator(x)
    