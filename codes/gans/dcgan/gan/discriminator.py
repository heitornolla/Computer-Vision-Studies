import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, img_channels, features):
        super(Discriminator, self).__init__()
        # Input = 64x64
        self.discriminator = nn.Sequential(
            nn.Conv2d(
                img_channels, features, kernel_size=4, stride=2, padding=1
            ), # 32x32
            nn.LeakyReLU(0.2), # No BatchNorm in initial layer
            self._block(features, features*2, 4, 2, 1), # 16x16
            self._block(features*2, features*4, 4, 2, 1), # 8x8
            self._block(features*4, features*8, 4, 2, 1), # 4x4
            nn.Conv2d(features*8, 1, 4, 2, 0), # 1x1
            nn.Sigmoid()
        )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        return self.discriminator(x)
