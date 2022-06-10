import argparse

from utils import *


def parse_args():
    desc = "AnimeGANv2"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--name', type=str, default='expr', 
                        help='experiment name')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--dataset', type=str, default='Hayao', 
                        help='dataset_name')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='The size of batch size')
    parser.add_argument('--save_freq', type=int, default=1,
                        help='The number of ckpt_save_freq')
    #---------------------------optimier--------------------------------------
    parser.add_argument('--lr_policy', type=str, default='step',
                        help="linear | step | consine")
    parser.add_argument('--init_lr', type=float, default=2e-4,
                        help='The learning rate')
    parser.add_argument('--g_lr', type=float, default=8e-5,
                        help='The learning rate')
    parser.add_argument('--d_lr', type=float, default=1.6e-4,
                        help='The learning rate')
    parser.add_argument('--g_beta1', type=float, default=0.5,
                        help='The learning rate')
    parser.add_argument('--d_beta1', type=float, default=0.5,
                        help='The learning rate')
    #----------------------------weights---------------------------------------
    parser.add_argument('--g_adv_weight', type=float, default=300.0, 
                        help='Weight about GAN')
    parser.add_argument('--d_adv_weight', type=float, default=300.0,
                        help='Weight about GAN')
    # 1.5 for Hayao, 2.0 for Paprika, 1.2 for Shinkai
    parser.add_argument('--con_weight', type=float, default=1.5,
                        help='Weight about VGG19')
    # ------ the follow weight used in AnimeGAN
    # 2.5 for Hayao, 0.6 for Paprika, 2.0 for Shinkai
    parser.add_argument('--sty_weight', type=float, default=2.5,
                         help='Weight about style')
    # 15. for Hayao, 50. for Paprika, 10. for Shinkai
    parser.add_argument('--color_weight', type=float, default=10.,
                         help='Weight about color')
    # 1. for Hayao, 0.1 for Paprika, 1. for Shinkai
    parser.add_argument('--tv_weight', type=float, default=1.,
                        help='Weight about tv')
    # -------------------------networkd---------------------------
    parser.add_argument('--gan_type', type=str, default='lsgan',
                        help='[gan / lsgan / wgan-gp / wgan-lp / dragan / hinge')

    parser.add_argument('--img_size', type=list, default=[256, 256],
                        help='The size of image: H and W')
    parser.add_argument('--img_ch', type=int, default=3,
                        help='The size of image channel')

    parser.add_argument('--ch', type=int, default=64,
                        help='base channel number per layer')
    parser.add_argument('--n_dis', type=int, default=3,
                        help='The number of discriminator layer')

    return check_args(parser.parse_args())


def check_args(args):
    """checking arguments"""
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)
    # --log_dir
    check_folder(args.log_dir)
    # --sample_dir
    check_folder(args.sample_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args


def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # main

    

if __name__ == '__main__':
    main()
