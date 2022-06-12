import torch
import torch.nn.functional as F
import torch.nn as nn
from networks import VGG19
from utils import gram, rgb_to_yuv, DEVICE


class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.huber = nn.SmoothL1Loss()

    def forward(self, image, image_g):
        image = rgb_to_yuv(image)
        image_g = rgb_to_yuv(image_g)

        # After convert to yuv, both images have channel last

        return (self.l1(image[:, :, :, 0], image_g[:, :, :, 0]) +
                self.huber(image[:, :, :, 1], image_g[:, :, :, 1]) +
                self.huber(image[:, :, :, 2], image_g[:, :, :, 2]))


class AnimeGanLoss:
    def __init__(self, args):
        self.content_loss = nn.L1Loss().to(DEVICE)
        self.gram_loss = nn.L1Loss().to(DEVICE)
        self.color_loss = ColorLoss().to(DEVICE)
        self.wadvg = args.g_adv_weight
        self.wadvd = args.d_adv_weight
        self.wcon = args.con_weight
        self.wgra = args.sty_weight
        self.wcol = args.color_weight
        self.vgg19 = VGG19().to(DEVICE).eval()
        self.adv_type = args.gan_loss
        self.bce_loss = nn.BCELoss()

    def compute_loss_G(self, fake_img, img, fake_logit, anime_gray):
        '''
        Compute loss for Generator

        @Arugments:
            - fake_img: generated image
            - img: image
            - fake_logit: output of Discriminator given fake image
            - anime_gray: grayscale of anime image

        @Returns:
            loss
        '''
        fake_feat = self.vgg19(fake_img)[3]
        anime_feat = self.vgg19(anime_gray)[3]
        img_feat = self.vgg19(img)[3]
        
        return [
            self.wadvg * self.adv_loss_g(fake_logit),
            self.wcon * self.content_loss(img_feat, fake_feat),
            self.wgra * self.gram_loss(gram(anime_feat), gram(fake_feat)),
            self.wcol * self.color_loss(img, fake_img),
        ]

    def compute_loss_D(self, fake_img_d, real_anime_d, real_anime_gray_d, real_anime_smooth_gray_d):
        return self.wadvd * (
            self.adv_loss_d_real(real_anime_d) +
            self.adv_loss_d_fake(fake_img_d) +
            self.adv_loss_d_fake(real_anime_gray_d) +
            0.2 * self.adv_loss_d_fake(real_anime_smooth_gray_d)
        )

    def content_loss_vgg(self, image, recontruction):
        feat = self.vgg19(image)[3]
        re_feat = self.vgg19(recontruction)[3]

        return self.content_loss(feat, re_feat)

    def adv_loss_d_real(self, pred):
        if self.adv_type == 'hinge':
            return torch.mean(F.relu(1.0 - pred))

        elif self.adv_type == 'lsgan':
            return torch.mean(torch.square(pred - 1.0))

        elif self.adv_type == 'normal':
            return self.bce_loss(pred, torch.ones_like(pred))

        raise ValueError(f'Do not support loss type {self.adv_type}')

    def adv_loss_d_fake(self, pred):
        if self.adv_type == 'hinge':
            return torch.mean(F.relu(1.0 + pred))

        elif self.adv_type == 'lsgan':
            return torch.mean(torch.square(pred))

        elif self.adv_type == 'normal':
            return self.bce_loss(pred, torch.zeros_like(pred))

        raise ValueError(f'Do not support loss type {self.adv_type}')

    def adv_loss_g(self, pred):
        if self.adv_type == 'hinge':
            return -torch.mean(pred)

        elif self.adv_type == 'lsgan':
            return torch.mean(torch.square(pred - 1.0))

        elif self.adv_type == 'normal':
            return self.bce_loss(pred, torch.zeros_like(pred))

        raise ValueError(f'Do not support loss type {self.adv_type}')


class LossSummary:
    def __init__(self):
        self.reset()

    def reset(self):
        self.loss_g_adv = []
        self.loss_content = []
        self.loss_gram = []
        self.loss_color = []
        self.loss_d_adv = []

    def update_loss_G(self, adv, gram, color, content):
        self.loss_g_adv.append(adv.cpu().detach().numpy())
        self.loss_gram.append(gram.cpu().detach().numpy())
        self.loss_color.append(color.cpu().detach().numpy())
        self.loss_content.append(content.cpu().detach().numpy())

    def update_loss_D(self, loss):
        self.loss_d_adv.append(loss.cpu().detach().numpy())

    def avg_loss_G(self):
        return (
            self._avg(self.loss_g_adv),
            self._avg(self.loss_gram),
            self._avg(self.loss_color),
            self._avg(self.loss_content),
        )

    def avg_loss_D(self):
        return self._avg(self.loss_d_adv)

    @staticmethod
    def _avg(losses):
        return sum(losses) / len(losses)
