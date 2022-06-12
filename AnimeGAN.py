import gc
import os
from collections import OrderedDict
import cv2

import torch
import torch.nn as nn
from fvcore.nn import parameter_count, parameter_count_table

from networks import Generator, Discriminator
from losses import AnimeGanLoss, LossSummary
from utils import DEVICE, save_checkpoint, set_lr, tensor2im


class AnimeGAN():
    def __init__(self, args):
        """Initialization steps."""
        self.args = args
        self.isTrain = args.isTrain
        print("Init models...")
        self.model_names = ['G', 'D']
        self.loss_names = ['adv', 'con', 'gra', 'col', 'd']
        self.visual_names = ['p', 'fake']
        self.netG = Generator(args)

        if self.isTrain:
            self.netD = Discriminator(args)
            self.loss_tracker = LossSummary()
            self.loss_fn = AnimeGanLoss(args)
            self.optimizer_g = torch.optim.Adam(self.netG.parameters(), lr=args.lr_g, betas=(args.lr_beta1_g, 0.999))
            self.optimizer_d = torch.optim.Adam(self.netD.parameters(), lr=args.lr_d, betas=(args.lr_beta1_d, 0.999))
    
    def setup(self):
        """Load and print networks"""
        if not self.isTrain:
            self.load_networks(self.args.load_iter)
        self.print_networks(self.args.verbose)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps."""
        self.p = input['p'].to(DEVICE)
        self.a = input['a'].to(DEVICE)
        self.x = input['x'].to(DEVICE)
        self.y = input['y'].to(DEVICE)

    #-----------------------------training------------------------------------
    
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake = self.netG(self.p)
    
    def backward_D(self):
        """Calculate GAN loss for discriminator D"""
        fake_d = self.netD(self.fake.detach())
        real_anime_d = self.netD(self.a)
        real_anime_gray_d = self.netD(self.x)
        real_anime_smt_gray_d = self.netD(self.y)
        loss_d = self.loss_fn.compute_loss_D(
            fake_d, real_anime_d, real_anime_gray_d, real_anime_smt_gray_d)
        loss_d.backward()
        self.loss_d = loss_d.cpu().detach().item()
        
    def backward_G(self):
        """Calculate the loss for generators G"""
        fake_d = self.netD(self.fake)
        adv_loss, con_loss, gra_loss, col_loss = self.loss_fn.compute_loss_G(
            self.fake, self.p, fake_d, self.x)
        loss_g = adv_loss + con_loss + gra_loss + col_loss
        loss_g.backward()
        self.loss_adv = adv_loss.cpu().detach().item()
        self.loss_gra = gra_loss.cpu().detach().item()
        self.loss_con = con_loss.cpu().detach().item()
        self.loss_col = col_loss.cpu().detach().item()

    def init_generator(self, bar):
        set_lr(self.optimizer_g, self.args.init_lr)
        self.optimizer_g.zero_grad()

        fake_img = self.netG(self.p)
        loss = self.loss_fn.content_loss_vgg(self.p, fake_img)
        loss.backward()
        self.optimizer_g.step()
        bar.set_description(
            f'[Init Training G] content loss: {loss:2f}')


    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images
        # Fix G, train D
        self.set_requires_grad([self.netD], True)
        self.optimizer_d.zero_grad()  # set D's gradients to zero
        self.backward_D()             # calculate graidents for D_B
        self.optimizer_d.step()       # update D_A and D_B's weights
        # Fix D, train G
        self.set_requires_grad([self.netD], False)
        self.optimizer_g.zero_grad()  # set G's gradients to zero
        self.backward_G()             # calculate gradients for G
        self.optimizer_g.step()       # update G weights
    
    #-----------------------------test------------------------------------
    
    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                net.eval()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()
            
    #-----------------------------log losses & imgs------------------------------------

    def get_current_visuals(self):
        """返回需要可视化的图像，需要在self.visual_names列表中声明"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """返回定义的各种loss，需要在self.loss_names列表中声明"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(
                    getattr(self, "loss_" + name)
                )  # float(...) works for both scalar tensor and float number
        return errors_ret

    #-----------------------------save & load------------------------------------

    def save_networks(self, iter_step):
        """保存模型"""
        save_checkpoint(self.netG, self.optimizer_g, iter_step,
                        self.args.checkpoint_dir, self.args.name, 'G')
        save_checkpoint(self.netD, self.optimizer_d, iter_step, 
                        self.args.checkpoint_dir, self.args.name, 'D')

    def load_networks(self, iter_step):
        """Load all the networks from the disk."""
        # 指定路径
        path = os.path.join(self.args.checkpoint_dir, f'G_{self.args.name}_{iter_step}.pth')
        # 读取存储的二进制对象
        checkpoint = torch.load(path,  map_location='cuda:0') if torch.cuda.is_available() else torch.load(path,  map_location='cpu')
        # 读取
        self.netG.load_state_dict(checkpoint['model'], strict=True)
        if self.isTrain:
            self.optimizer_g.load_state_dict(checkpoint['optim'], strict=True)
            self.netD.load_state_dict(checkpoint['model'], strict=True)
            self.optimizer_d.load_state_dict(checkpoint['optim'], strict=True)
        del checkpoint
        torch.cuda.empty_cache()
        gc.collect()
        print('network loaded!!!!!')

    def print_networks(self, verbose):
        """打印网络的参数总量

        Parameters:
            verbose (bool) -- if verbose: print the network parmeters per layer
        """
        print("---------- Networks initialized -------------")
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                num_params = parameter_count(net)['']
                if verbose:
                    print(parameter_count_table(net))
                print(
                    "[Network %s] Total number of parameters : %.3f M"
                    % (name, num_params / 1e6)
                )
        print("-----------------------------------------------")

    def set_requires_grad(self, nets, requires_grad=False):
        """冻结/解冻指定网络"""
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def save_samples(self, prefix='gen'):
        '''Generate and save images'''
        for i, img in enumerate(self.fake.detach().cpu().unbind()):
            save_path = os.path.join(self.args.save_image_dir, f'{prefix}_{i}.jpg')
            cv2.imwrite(save_path, tensor2im(img))
