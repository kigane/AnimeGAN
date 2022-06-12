import yaml
import argparse
from utils import *


def parse_args():
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
    check_folder(args.checkpoints)
    check_folder(args.log_dir)

    # check datasets
    if not os.path.exists(args.datarootA):
        raise FileNotFoundError(f'Dataset not found {args.datarootA}')
    if not os.path.exists(args.datarootB):
        raise FileNotFoundError(f'Dataset not found {args.datarootB}')

    assert args.gan_loss in {'lsgan', 'hinge', 'bce'}, f'{args.gan_loss} is not supported'
    return args


if __name__ == '__main__':
    parse_args()
