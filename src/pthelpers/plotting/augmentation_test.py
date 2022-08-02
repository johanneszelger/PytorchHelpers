import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch import Tensor
from torchvision.transforms import ToPILImage

try:
    from pthelpers.utils.vision_transforms_factory import generate_train_transforms, vision_transforms_factory_ingredient
except:
    from src.pthelpers.utils.vision_transforms_factory import generate_train_transforms, vision_transforms_factory_ingredient

from sacred import Experiment
from torch.utils.data import DataLoader

plt.rcParams["savefig.bbox"] = 'tight'
torch.manual_seed(0)


def show_transforms(dataloader: DataLoader, num_originals: int = 1, num_augs: int = 3):
    trafos = generate_train_transforms()
    for i, (x, y) in enumerate(dataloader):
        if i == num_originals: break
        if isinstance(x, Tensor): x = ToPILImage()(x)
        aug = [ToPILImage()(trafos(x)) for _ in range(num_augs)]

        __plot__(x, aug)


def __plot__(orig, imgs, with_orig=True, row_title=None, **imshow_kwargs):
    if (with_orig): imgs = [orig] + imgs

    num_cols = math.sqrt(len(imgs))
    num_cols = num_cols if num_cols % 1 == 0 else math.floor(num_cols) + 1
    num_cols = int(num_cols)
    num_rows = num_cols

    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(imgs[row_idx * num_cols + col_idx]), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    ex = Experiment("test", ingredients=[vision_transforms_factory_ingredient])
    vision_transforms_factory_ingredient.add_config({
        "resize": {
            "width": 200,
            "height": 200
        },
        "rotate": 90
    })


    @ex.automain
    def main():
        data = torchvision.datasets.MNIST("./data/", download=True)
        show_transforms(data)
