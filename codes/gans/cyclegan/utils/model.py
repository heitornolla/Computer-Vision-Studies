import torch
import torch.nn as nn
import torch.nn.functional as F

# helper conv function
def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, normalization=True, norm_type='instance_norm'):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
    if normalization:
        if norm_type == 'instance_norm':
            layers.append(nn.InstanceNorm2d(out_channels))
        elif norm_type == 'batch_norm':
            layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)

# helper deconv function
def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, normalization=True, norm_type='instance_norm'):
    layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
    if normalization:
        if norm_type == 'instance_norm':
            layers.append(nn.InstanceNorm2d(out_channels))
        elif norm_type == 'batch_norm':
            layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)

class ResidualBlock(nn.Module):
    def __init__(self, conv_dim):
        super().__init__()
        self.conv_layer1 = conv(conv_dim, conv_dim, 3, stride=1, padding=1, normalization=True)
        self.conv_layer2 = conv(conv_dim, conv_dim, 3, stride=1, padding=1, normalization=True)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reset_parameters()

    @staticmethod
    def init_weights(layer):
        if isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)

    def reset_parameters(self):
        self.conv_layer1.apply(self.init_weights)
        self.conv_layer2.apply(self.init_weights)

    def forward(self, x):
        out_1 = self.leaky_relu(self.conv_layer1(x))
        out_2 = x + self.conv_layer2(out_1)
        return out_2

class GlobalDiscriminator(nn.Module):
    def __init__(self, conv_dim=64):
        super().__init__()
        self.image_size = (1024, 256)
        self.conv1 = conv(3, conv_dim, 4, normalization=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        self.conv4 = conv(conv_dim*4, conv_dim*4, 4)
        self.conv5 = conv(conv_dim*4, conv_dim*4, 4)
        self.conv6 = conv(conv_dim*4, conv_dim*4, 4)
        self.conv7 = conv(conv_dim*4, conv_dim*4, 4)
        self.conv8 = conv(conv_dim*4, conv_dim*4, 4)
        fc_in_size = int(conv_dim*4*self.image_size[0]*self.image_size[1] / ((2**8)*(2**8)))
        self.fc1 = nn.Linear(fc_in_size, 64)
        self.disc_out = nn.Linear(64, 1)
        self.flatten = nn.Flatten()
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reset_parameters()

    @staticmethod
    def init_weights(layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(layer.weight)

    def reset_parameters(self):
        for module in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8, self.fc1, self.disc_out]:
            module.apply(self.init_weights)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        out = F.relu(self.conv6(out))
        out = F.relu(self.conv7(out))
        out = F.relu(self.conv8(out))
        out = self.flatten(out)
        out = F.relu(self.fc1(out))
        out = self.disc_out(out)
        return out

class PatchDiscriminator(nn.Module):
    def __init__(self, conv_dim=64):
        super().__init__()
        self.conv1 = conv(3, conv_dim, 4, normalization=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        self.conv4 = conv(conv_dim*4, conv_dim*8, 4)
        self.conv_final = conv(conv_dim*8, 1, 3, stride=1, normalization=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reset_parameters()

    @staticmethod
    def init_weights(layer):
        if isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)

    def reset_parameters(self):
        for module in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv_final]:
            module.apply(self.init_weights)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = self.conv_final(out)
        return out

class Generator(nn.Module):
    def __init__(self, conv_dim=64, n_res_blocks=6):
        super().__init__()
        self.conv1 = conv(3, conv_dim, 4)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        self.conv4 = conv(conv_dim*4, conv_dim*4, 4)
        self.res_blocks = nn.Sequential(*[ResidualBlock(conv_dim*4) for _ in range(n_res_blocks)])
        self.deconv1 = deconv(conv_dim*4, conv_dim*4, 4)
        self.deconv2 = deconv(conv_dim*4, conv_dim*2, 4)
        self.deconv3 = deconv(conv_dim*2, conv_dim, 4)
        self.out_layer = deconv(conv_dim, 3, 4, normalization=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reset_parameters()

    @staticmethod
    def init_weights(layer):
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_uniform_(layer.weight)

    def reset_parameters(self):
        for module in [self.conv1, self.conv2, self.conv3, self.conv4, self.deconv1, self.deconv2, self.deconv3, self.out_layer]:
            module.apply(self.init_weights)

    def forward(self, x):
        out = self.leaky_relu(self.conv1(x))
        out = self.leaky_relu(self.conv2(out))
        out = self.leaky_relu(self.conv3(out))
        out = self.leaky_relu(self.conv4(out))
        out = self.res_blocks(out)
        out = self.leaky_relu(self.deconv1(out))
        out = self.leaky_relu(self.deconv2(out))
        out = self.leaky_relu(self.deconv3(out))
        out = torch.tanh(self.out_layer(out))
        out = torch.clamp(out, min=-0.5, max=0.5)
        return out

def CycleGAN(g_conv_dim=64, d_conv_dim=64, n_res_blocks=6):
    G_XtoY = Generator(conv_dim=g_conv_dim, n_res_blocks=n_res_blocks)
    G_YtoX = Generator(conv_dim=g_conv_dim, n_res_blocks=n_res_blocks)
    Dp_X = PatchDiscriminator(conv_dim=d_conv_dim)
    Dp_Y = PatchDiscriminator(conv_dim=d_conv_dim)
    Dg_X = GlobalDiscriminator(conv_dim=d_conv_dim)
    Dg_Y = GlobalDiscriminator(conv_dim=d_conv_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for model in [G_XtoY, G_YtoX, Dp_X, Dp_Y, Dg_X, Dg_Y]:
        model.to(device)

    print(f'Using {"GPU" if torch.cuda.is_available() else "CPU"}.')
    return G_XtoY, G_YtoX, Dp_X, Dp_Y, Dg_X, Dg_Y

