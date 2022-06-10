import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


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
    _, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.reshape(-1)
    for i in range(rows * cols):
        axes[i].imshow(tensor2im(tensors[i]))
        axes[i].set_axis_off()
    plt.show()

