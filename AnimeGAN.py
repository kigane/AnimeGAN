import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from fvcore.nn import parameter_count, parameter_count_table

import networks


class AnimeGANv2():
    def __init__(self, opt):
        """Initialization steps. you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         define networks used in our training.
            -- self.visual_names (str list):        specify the images that you want to display and save.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them.
        """
        self.opt = opt
        self.isTrain = opt.isTrain
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.loss_names = ['G', 'D', 'con', 'gray', 'col']
        self.visual_names = ['p', 'a', 'x', 'y', 'ret']
        self.optimizers = []
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']

        self.netG = networks.define_G(
            opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.init_type, opt.init_gain
        )
        
        if self.isTrain:
            self.netD = networks.define_D(
                opt.output_nc, opt.ngf, opt.netG, opt.init_type, opt.init_gain
            )
            # define loss functions
            # define GAN loss.
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionContent = nn.L1Loss()
            self.criterionGray = nn.L1Loss()
            self.criterionColorY = nn.L1Loss()
            self.criterionColorUV = nn.HuberLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.g_lr, betas=(opt.g_beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.d_lr, betas=(opt.d_beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
    
    def setup(self, opt):
        """Load and print networks; create schedulers"""
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)
        self.print_networks(opt.verbose)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps."""
        pass
    
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass
    
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
    
    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def update_learning_rate(self):
        """为所有网络更新学习率; 在每个epoch的末尾调用"""
        for scheduler in self.schedulers:
            if self.opt.lr_policy == "plateau":
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]["lr"]
        print("learning rate = %.7f" % lr)

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

    def save_networks(self, iter_step):
        """保存self.model_names列表中标明的模型到save_dir"""
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = f"{iter_step}_net_{name}.pth"
                save_path = os.path.join(
                    self.save_dir, "models", save_filename)
                net = getattr(self, "net" + name)

                if torch.cuda.is_available():
                    torch.save(net.cpu().state_dict(), save_path)
                    net.cuda()
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def load_networks(self, iter_step):
        """Load all the networks from the disk."""
        self.load_pretrain_from_path(self.save_dir, iter_step)

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
