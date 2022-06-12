from tqdm import tqdm
from AnimeGAN import AnimeGAN
from dataset import create_data_iter
from utils import *


def train():
    args = parse_args()

    # prepare dataset
    data_iter = create_data_iter(args)
    
    # model initialization
    model = AnimeGAN(args)
    model.setup()
    if args.use_wandb:
        visual_table = wandb.Table(columns=["iter" ,"photo", "fake"])

    # trainning loop
    for step in (pbar := tqdm(range(1, args.iterations+1))):
        # training
        batch_data = next(data_iter).to(DEVICE)
        # unpack data from dataset and apply preprocessing
        model.set_input(batch_data)
        if step <= args.init_iters:
            if step == 1:
                set_lr(model.optimizer_g, args.lr_init)
            model.init_generator(pbar)
            if step == args.init_iters:
                set_lr(model.optimizer_g, args.lr_g)
                model.save_networks(step)
                model.save_samples(prefix='init')
            continue
        else:
        # calculate loss functions, get gradients, update network weights
            model.optimize_parameters()

        # show and log training information
        # losses
        desc = ''
        loss_dict = model.get_current_losses()
        for k,v in loss_dict.items():
            desc += f'{k}={v:.3f}'
        pbar.set_description(desc)

        if args.use_wandb:
            wandb.log(loss_dict, step)

        # model state
        if step % args.save_freq == 0:
            visual_dict = model.get_current_visuals()
            for i in range(args.batch_size):
                visual_table.add_data(
                    step,
                    wandb.Image(tensor2im(visual_dict['p'][i])),
                    wandb.Image(tensor2im(visual_dict['fake'][i])),
                )
            wandb.log('visual', visual_table)
            model.save_networks(step)
            model.save_samples()


if __name__ == '__main__':
    train()
