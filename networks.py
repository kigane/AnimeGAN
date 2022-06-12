from utils import DEVICE, initialize_weights
from conv_blocks import ConvBlock
from conv_blocks import InvertedResBlock
from conv_blocks import SeparableConv2D
from conv_blocks import UpConv
from conv_blocks import DownConv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn import init
from torch.nn.utils import spectral_norm
from torch.optim import lr_scheduler

# -------------------------Helper Functions-----------------------------------


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler"""
    if opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=opt.iter_step, eta_min=0)
    else:
        return NotImplementedError(f'learning rate policy [{opt.lr_policy}] is not implemented')
    return scheduler


def init_weights(
    net,
    init_type='normal',  # initialization method: normal | xavier | kaiming | orthogonal
    init_gain=0.02      # scaling factor for normal, xavier and orthogonal.
):
    """Initialize network weights."""
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print(f'initialize network with {init_type}')
    net.apply(init_func)  # apply the initialization function <init_func>

# -------------------------Classes--------------------------------------------

class Generator(nn.Module):
    def __init__(self, dataset=''):
        super(Generator, self).__init__()
        self.name = f'generator_{dataset}'
        bias = False

        self.encode_blocks = nn.Sequential(
            ConvBlock(3, 64, bias=bias),
            ConvBlock(64, 128, bias=bias),
            DownConv(128, bias=bias),
            ConvBlock(128, 128, bias=bias),
            SeparableConv2D(128, 256, bias=bias),
            DownConv(256, bias=bias),
            ConvBlock(256, 256, bias=bias),
        )

        self.res_blocks = nn.Sequential(
            InvertedResBlock(256, 256, bias=bias),
            InvertedResBlock(256, 256, bias=bias),
            InvertedResBlock(256, 256, bias=bias),
            InvertedResBlock(256, 256, bias=bias),
            InvertedResBlock(256, 256, bias=bias),
            InvertedResBlock(256, 256, bias=bias),
            InvertedResBlock(256, 256, bias=bias),
            InvertedResBlock(256, 256, bias=bias),
        )

        self.decode_blocks = nn.Sequential(
            ConvBlock(256, 128, bias=bias),
            UpConv(128, bias=bias),
            SeparableConv2D(128, 128, bias=bias),
            ConvBlock(128, 128, bias=bias),
            UpConv(128, bias=bias),
            ConvBlock(128, 64, bias=bias),
            ConvBlock(64, 64, bias=bias),
            spectral_norm(nn.Conv2d(64, 3, kernel_size=1,
                          stride=1, padding=0, bias=bias)),
            nn.Tanh(),
        )

        initialize_weights(self)

    def forward(self, x):
        out = self.encode_blocks(x)
        out = self.res_blocks(out)
        img = self.decode_blocks(out)

        return img


class Discriminator(nn.Module):
    def __init__(self,  args):
        super(Discriminator, self).__init__()
        self.name = f'discriminator_{args.name}'
        self.bias = False
        channels = 32

        layers = [
            nn.Conv2d(3, channels, kernel_size=3,
                      stride=1, padding=1, bias=self.bias),
            nn.LeakyReLU(0.2, True)
        ]

        for i in range(args.d_layers):
            layers += [
                nn.Conv2d(channels, channels * 2, kernel_size=3,
                          stride=2, padding=1, bias=self.bias),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(channels * 2, channels * 4, kernel_size=3,
                          stride=1, padding=1, bias=self.bias),
                nn.InstanceNorm2d(channels * 4),
                nn.LeakyReLU(0.2, True),
            ]
            channels *= 4

        layers += [
            nn.Conv2d(channels, channels, kernel_size=3,
                      stride=1, padding=1, bias=self.bias),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channels, 1, kernel_size=3,
                      stride=1, padding=1, bias=self.bias),
        ]

        if args.use_sn:
            for i in range(len(layers)):
                if isinstance(layers[i], nn.Conv2d):
                    layers[i] = spectral_norm(layers[i])

        self.discriminate = nn.Sequential(*layers)

        initialize_weights(self)

    def forward(self, img):
        return self.discriminate(img)

# ----------------------------------------------------------------------------

class VGG19(torch.nn.Module):
    """VGG architecter, used for the perceptual loss using a pretrained VGG network"""
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(
            pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 32):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        for x in range(32, 36):
            self.slice6.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)

        # self.mean = torch.tensor([0.485, 0.456, 0.406]).view(
        #     1, -1, 1, 1).to(DEVICE) * 2 - 1
        # self.std = torch.tensor([0.229, 0.224, 0.225]).view(
        #     1, -1, 1, 1).to(DEVICE) * 2

    def forward(self, X):  # relui_1
        # X = (X-self.mean)/self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5[:-2](h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    """Perceptual loss that uses a pretrained VGG network"""
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().to(DEVICE)
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss
