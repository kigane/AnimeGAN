import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from utils import initialize_weights

class Discriminator(nn.Module):
    # initializers
    def __init__(self, in_nc, out_nc, nf=32):
        super(Discriminator, self).__init__()
        self.input_nc = in_nc
        self.output_nc = out_nc
        self.nf = nf
        self.convs = nn.Sequential(
            spectral_norm(nn.Conv2d(in_nc, nf, 3, 1, 1, bias=False)),
            nn.LeakyReLU(0.2, True),

            spectral_norm(nn.Conv2d(nf, nf * 2, 3, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False)),
            nn.InstanceNorm2d(nf * 4),
            nn.LeakyReLU(0.2, True),

            spectral_norm(nn.Conv2d(nf * 4, nf * 4, 3, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(nf * 4, nf * 8, 3, 1, 1, bias=False)),
            nn.InstanceNorm2d(nf * 8),
            nn.LeakyReLU(0.2, True),

            spectral_norm(nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False)),
            nn.InstanceNorm2d(nf * 8),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(nf * 8, out_nc, 3, 1, 1, bias=False)),
            # nn.Sigmoid(), # 原始 GAN Loss
        )

        initialize_weights(self)

    # forward method
    def forward(self, input):
        output = self.convs(input)

        return output
