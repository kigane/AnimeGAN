import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
import yaml

#----------------------------------------------------------------------------

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#---------------------handle args---------------------------------------

def parse_args():
    """read args from two config files specified by --basic and --advance"""
    desc = "AnimeGAN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--basic', type=str,
                        default='config/basic.yml', help='basic options')
    parser.add_argument('--advance', type=str,
                        default='config/config.yml', help='model options')
    return check_args(parser.parse_args())


def check_args(args):
    """combine arguments"""
    with open(args.basic, 'r') as f:
        basic_config = yaml.safe_load(f)
    with open(args.advance, 'r') as f:
        advance_config = yaml.safe_load(f)
    args_dict = vars(args)
    args_dict.update(basic_config)
    args_dict.update(advance_config)

    # check dirs
    check_folder(args.checkpoint_dir)
    check_folder(args.log_dir)

    # check datasets
    if not os.path.exists(args.content_dir):
        raise FileNotFoundError(f'Dataset not found {args.content_dir}')
    if not os.path.exists(args.style_dir):
        raise FileNotFoundError(f'Dataset not found {args.style_dir}')

    assert args.gan_loss in {'lsgan', 'hinge',
                             'bce'}, f'{args.gan_loss} is not supported'
    
    if args.use_wandb:
        wandb.init(
            project=args.project,
            group=args.group,
            notes=args.notes,
            config=advance_config
        )
    return args


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        print(f'* {log_dir} does not exist, creating...')
        os.makedirs(log_dir)
    return log_dir

#-------------------------show and save images---------------------------------------

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array (1, 3, h, w) or (3, h, w) into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        # convert it into a numpy array
        if image_tensor.dim() == 4:
            image_numpy = image_tensor[0].cpu().float().numpy()
        else:
            image_numpy = image_tensor.cpu().float().numpy()
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        # post-processing: tranpose and scaling
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  
        image_numpy = np.clip(image_numpy, 0, 255)
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def show_tensor_imgs(tensors, rows=1, cols=1):
    """show a batch of images"""
    _, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axes = axes.reshape(-1)
    for i in range(rows * cols):
        axes[i].imshow(tensor2im(tensors[i]))
        axes[i].set_axis_off()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

#------------------------------losses-----------------------------------------

_rgb_to_yuv_kernel = torch.tensor([
    [0.299, -0.14714119, 0.61497538],
    [0.587, -0.28886916, -0.51496512],
    [0.114, 0.43601035, -0.10001026]
]).float()

if torch.cuda.is_available():
    _rgb_to_yuv_kernel = _rgb_to_yuv_kernel.cuda()


def gram(input):
    """
    Calculate Gram Matrix

    https://pytorch.org/tutorials/advanced/neural_style_tutorial.html#style-loss
    """
    b, c, w, h = input.size()
    x = input.view(b * c, w * h)
    G = torch.mm(x, x.T)
    # normalize by total elements
    return G.div(b * c * w * h)


def rgb_to_yuv(image):
    '''
    https://en.wikipedia.org/wiki/YUV

    output: Image of shape (H, W, C) (channel last)
    '''
    # -1 1 -> 0 1
    image = (image + 1.0) / 2.0

    yuv_img = torch.tensordot(
        image,
        _rgb_to_yuv_kernel,
        dims=([image.ndim - 3], [0]))

    return yuv_img

#----------------------------------------------------------------------------

def initialize_weights(net):
    for m in net.modules():
        try:
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        except Exception as e:
            # print(f'SKip layer {m}, {e}')
            pass


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(model, optimizer, iter_step, dir, name, prefix=''):
    checkpoint = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'iterations': iter_step
    }
    path = os.path.join(dir, f'{prefix}_{name}_{iter_step}.pth')
    torch.save(checkpoint, path)

