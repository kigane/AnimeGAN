import os
import random
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision.transforms import transforms
from functools import partial


def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class InfiniteSamplerWrapper(Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31


class FlatFolderDataset(Dataset):
    def __init__(self, content_dir, style_dir, transform, transform_G):
        super(FlatFolderDataset, self).__init__()
        self.c_fullpath = partial(os.path.join, content_dir)
        self.a_fullpath = partial(os.path.join, os.path.join(style_dir, 'style'))
        self.s_fullpath = partial(os.path.join, os.path.join(style_dir, 'smooth'))
        self.content_paths = os.listdir(content_dir)
        self.style_paths = os.listdir(os.path.join(style_dir, 'style'))
        self.smooth_paths = os.listdir(os.path.join(style_dir, 'smooth'))
        self.c_size = len(self.content_paths)
        self.a_size = len(self.style_paths)
        self.transform = transform
        self.transform_G = transform_G

    def __getitem__(self, index):
        c_path = self.content_paths[index % self.c_size]
        index_s = random.randint(0, self.a_size - 1)
        a_path = self.style_paths[index_s]
        s_path = self.smooth_paths[index_s]
        c_img = Image.open(self.c_fullpath(c_path)).convert('RGB')
        a_img = Image.open(self.a_fullpath(a_path)).convert('RGB')
        s_img = Image.open(self.s_fullpath(s_path)).convert('RGB')
        c = self.transform(c_img)   # photo
        a = self.transform(a_img)   # anime
        x = self.transform_G(a_img) # grayscale anime
        y = self.transform_G(s_img) # smoothed anime grayscale
    
        return c, a, x, y

    def __len__(self):
        return max(self.c_size, self.a_size)

    def name(self):
        return 'FlatFolderDataset'


def get_data_iter(args):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_G = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = FlatFolderDataset('./data/val', './data/Shinkai', transform, transform_G)
    dataloader = DataLoader(
        dataset, 
        batch_size=2, 
        sampler=InfiniteSamplerWrapper(dataset), 
        num_workers=0
    )
    return iter(dataloader)


if __name__ == '__main__':
    it = get_data_iter(None)
    b = next(it)
    print(len(b))
    from utils import inverse_transform, tensor2im
    import matplotlib.pyplot as plt
    print(b[3][0].data.shape)
    print(type(b[3][0]))
    img = tensor2im(b[3][0])
    print(img.shape)
    plt.imshow(img)
    plt.show()