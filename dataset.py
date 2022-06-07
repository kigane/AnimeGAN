import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from functools import partial

class MyDataset(Dataset):
    def __init__(self, content_dir, style_dir):
        super().__init__()
        self.content_img_path = partial(os.path.join, content_dir)
        self.style_img_path = partial(os.path.join, style_dir)
        self.content_img_list = os.listdir(content_dir)
        self.style_img_list = os.listdir(style_dir)
        