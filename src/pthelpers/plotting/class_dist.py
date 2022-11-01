import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader


def plot_class_dist(dl: DataLoader, n_classes: int, iter: int = 1, force_original_data: bool = False):
    counts = torch.zeros(n_classes)

    for i in range(iter):
        source = dl.dataset if force_original_data else dl
        for data in source:
            label = data[1]
            if isinstance(label, int):
                counts[label] += 1
            elif label.ndim == 1:
                counts += label.bincount(minlength=n_classes)
            else:
                label = label.argmax(dim=-1)
                if label.ndim == 1:
                    counts[label] += 1
                else:
                    counts += label.reshape(len(label)).bincount(minlength=10)

    plt.bar(np.arange(0, n_classes), counts.numpy())
    plt.title("original dist" if force_original_data else "sampled dist")
    plt.xlabel("Classes")
    plt.ylabel("Count")
    wandb.log({"original dist" if force_original_data else "sampled dist": plt.figure})
