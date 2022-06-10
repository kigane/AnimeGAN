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


def define_G(
    input_nc,           # the number of channels in input images
    output_nc,          # the number of channels in output images
    ngf,                # the number of filters in the last conv layer
    netG,               # the architecture's name: animegan
    init_type='normal',  # the name of our initialization method
    init_gain=0.02,     # scaling factor for normal, xavier and orthogonal.
):
    """Create a generator"""
    net = None

    if netG == 'animegan':
        net = Generator(
            input_nc, output_nc, ngf)
    else:
        raise NotImplementedError(
            'Generator model name [%s] is not recognized' % netG)
    return init_weights(net, init_type, init_gain)


def define_D(
    input_nc,            # the number of channels in input images
    ndf,                 # the number of filters in the first conv layer
    netD,                # the architecture's name: basic
    init_type='normal',  # the name of the initialization method.
    init_gain=0.02       # scaling factor for normal, xavier and orthogonal.
):    
    """Create a discriminator"""
    net = None

    if netD == 'basic':  # default PatchGAN classifier
        net = Discriminator(input_nc, 1, nf=ndf)
    else:
        raise NotImplementedError(
            'Discriminator model name [%s] is not recognized' % netD)
    return init_weights(net, init_type, init_gain)

# -------------------------Classes--------------------------------------------


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError(f'gan mode {gan_mode} not implemented')

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input."""
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels."""
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        return loss


class ConvNormLReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, pad_mode="reflect", groups=1, bias=False):

        pad_layer = {
            "zero":    nn.ZeroPad2d,
            "same":    nn.ReplicationPad2d,
            "reflect": nn.ReflectionPad2d,
        }
        if pad_mode not in pad_layer:
            raise NotImplementedError

        super(ConvNormLReLU, self).__init__(
            pad_layer[pad_mode](padding),
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                      stride=stride, padding=0, groups=groups, bias=bias),
            nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )


class InvertedResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expansion_ratio=2):
        super(InvertedResBlock, self).__init__()

        self.use_res_connect = in_ch == out_ch
        bottleneck = int(round(in_ch*expansion_ratio))
        layers = []
        # ConvBlock
        if expansion_ratio != 1:
            layers.append(ConvNormLReLU(
                in_ch, bottleneck, kernel_size=1, padding=0))

        # deepwise conv
        layers.append(ConvNormLReLU(bottleneck, bottleneck,
                      groups=bottleneck, bias=True))
        # pw-改进？
        layers.append(nn.Conv2d(bottleneck, out_ch,
                      kernel_size=1, padding=0, bias=False))
        layers.append(nn.GroupNorm(
            num_groups=1, num_channels=out_ch, affine=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        out = self.layers(input)
        if self.use_res_connect:
            out = input + out
        return out


class Generator(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.block_a = nn.Sequential(
            ConvNormLReLU(3,  32, kernel_size=7, padding=3),
            ConvNormLReLU(32, 64, stride=2, padding=(0, 1, 0, 1)),
            ConvNormLReLU(64, 64)
        )

        self.block_b = nn.Sequential(
            ConvNormLReLU(64,  128, stride=2, padding=(0, 1, 0, 1)),
            ConvNormLReLU(128, 128)
        )

        self.block_c = nn.Sequential(
            ConvNormLReLU(128, 128),
            InvertedResBlock(128, 256, 2),
            InvertedResBlock(256, 256, 2),
            InvertedResBlock(256, 256, 2),
            InvertedResBlock(256, 256, 2),
            ConvNormLReLU(256, 128),
        )

        self.block_d = nn.Sequential(
            ConvNormLReLU(128, 128),
            ConvNormLReLU(128, 128)
        )

        self.block_e = nn.Sequential(
            ConvNormLReLU(128, 64),
            ConvNormLReLU(64,  64),
            ConvNormLReLU(64,  32, kernel_size=7, padding=3)
        )

        self.out_layer = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

    def forward(self, input, align_corners=True):
        out = self.block_a(input)
        half_size = out.size()[-2:]
        out = self.block_b(out)
        out = self.block_c(out)

        if align_corners:
            out = F.interpolate(
                out, half_size, mode="bilinear", align_corners=True)
        else:
            out = F.interpolate(out, scale_factor=2,
                                mode="bilinear", align_corners=False)
        out = self.block_d(out)

        if align_corners:
            out = F.interpolate(
                out, input.size()[-2:], mode="bilinear", align_corners=True)
        else:
            out = F.interpolate(out, scale_factor=2,
                                mode="bilinear", align_corners=False)
        out = self.block_e(out)

        out = self.out_layer(out)
        return out


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
        )

        init_weights(self)

    # forward method
    def forward(self, input):
        output = self.convs(input)

        return output

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

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(
            1, -1, 1, 1).cuda() * 2 - 1
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(
            1, -1, 1, 1).cuda() * 2

    def forward(self, X):  # relui_1
        X = (X-self.mean)/self.std
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
        self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * \
                self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss
