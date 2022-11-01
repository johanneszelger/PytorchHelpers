import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets
from torchvision.transforms import ToTensor


def plot_class_dist(dl: DataLoader, n_classes: int, iter: int = 1, batch_size: int = 2048):
    counts_source = torch.zeros(n_classes)
    counts_sampled = torch.zeros(n_classes)

    orig_trafo = dl.dataset.transform


    def no_data(input):
        return torch.tensor([0])


    dl.dataset.transform = no_data
    for i in range(iter):
        counts_source += count_source(n_classes, DataLoader(dl.dataset, batch_size=batch_size, num_workers=16))
        counts_sampled += count_source(n_classes, dl)
    dl.dataset.transform = orig_trafo

    fig = plot_counts(counts_source, n_classes, True)
    fig2 = plot_counts(counts_sampled, n_classes, False)

    o_cols = ["c_" + str(i) for i in range(n_classes)]

    dist_table = wandb.Table(columns=o_cols)
    dist_table.add_data(*counts_source)
    dist_table.add_data(*counts_sampled)

    fig_table = wandb.Table(data=[[wandb.Image(fig), wandb.Image(fig2)]], columns=["original data", "sampled data"])
    wandb.log({"data distribution table": dist_table, "data distribution plots": fig_table})

def plot_counts(counts: torch.Tensor, n_classes: int, original: bool):
    fig, ax = plt.subplots()
    ax.bar(np.arange(0, n_classes), counts.numpy())
    ax.set_title("original data" if original else "sampled data")
    ax.set_xlabel("Classes")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def count_source(n_classes: int, source):
    counts = torch.zeros(n_classes)
    for data in source:
        label = data[1].int()
        if isinstance(label, int):
            counts[label] += 1
        elif label.ndim == 1:
            counts += label.bincount(minlength=n_classes)
        else:
            label = label.argmax(dim=-1)
            if len(label) == 1:
                counts[label] += 1
            else:
                counts += label.reshape(len(label)).bincount(minlength=n_classes)
    return counts


if __name__ == '__main__':
    wandb.init(project="dev")

    train_data = datasets.MNIST('../data', train=True, download=True)

    weights = [0.1, 1, 2, 3, 4, 3, 2, 1, 0.1, 1]
    samples_weight = [weights[t] for t in train_data.targets]
    sampler = WeightedRandomSampler(weights, 10000)
    loader = DataLoader(train_data, batch_size=512, sampler=sampler)

    plot_class_dist(loader, 10)
