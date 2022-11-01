import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets
from torchvision.transforms import ToTensor


def plot_samples(dl: DataLoader, n_classes: int, data_name: str ="training"):
    dataset = dl.dataset
    n_cols = wandb.config["tbl_img_per_row"] if "tbl_img_per_row" in wandb.config else 5
    tbl = wandb.Table(columns=["sample_" + str(i) for i in range(n_cols)])

    cls_names = wandb.config["class_names"] if "class_names" in wandb.config else np.arange(n_classes)

    g = torch.Generator()
    g.manual_seed(42)
    for i in range(n_classes):
        weights = np.zeros(n_classes)
        weights[i] = 1
        sample_weights = [weights[i] for i in dataset.targets]
        sampler = WeightedRandomSampler(sample_weights, 10000, generator=g)
        loader = DataLoader(dataset, batch_size=n_cols, sampler=sampler)

        for data in loader:
            imgs = data[0]

            tbl.add_data(cls_names[i], *[wandb.Image(img) for img in imgs])
            break

    wandb.log({f"sample {data_name}images": tbl})




if __name__ == '__main__':
    wandb.init(project="dev")

    train_data = datasets.MNIST('../data', train=True, download=True, transform=ToTensor())

    loader = DataLoader(train_data, batch_size=512)

    plot_class_dist(loader, 10)
