from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets
from torchvision.transforms import ToTensor

from pthelpers.models import SimpleNet
from pthelpers.training import Trainer
from pthelpers.utils.class_names import get_class_names


def confusion_matrix(
    trainer: Trainer, table_name: str,
    y_true: Optional[Sequence] = None,
    preds: Optional[Sequence] = None,
    class_names: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    panel: str = None
):

    assert len(preds) == len(
        y_true
    ), "Number of predictions and label indices must match"

    if class_names is not None:
        n_classes = len(class_names)
        class_inds = [i for i in range(n_classes)]
        assert max(preds) <= len(
            class_names
        ), "Higher predicted index than number of classes"
        assert max(y_true) <= len(
            class_names
        ), "Higher label class index than number of classes"
    else:
        class_inds = set(preds).union(set(y_true))
        n_classes = len(class_inds)
        class_names = [f"Class_{i}" for i in range(1, n_classes + 1)]

    # get mapping of inds to class index in case user has weird prediction indices
    class_mapping = {}
    for i, val in enumerate(sorted(list(class_inds))):
        class_mapping[val] = i
    counts = np.zeros((n_classes, n_classes))
    for i in range(len(preds)):
        counts[class_mapping[y_true[i]], class_mapping[preds[i]]] += 1

    data = []
    for i in range(n_classes):
        for j in range(n_classes):
            data.append([trainer.sample, class_names[i], class_names[j], counts[i, j]])


    fields = {
        "Actual": "Actual",
        "Predicted": "Predicted",
        "nPredictions": "nPredictions",
    }

    data = []
    for i in range(n_classes):
        for j in range(n_classes):
            data.append([trainer.sample, class_names[i], class_names[j], counts[i, j]])

    return trainer.wandb_log({f"{panel + '/' if panel is not None else ''}test": wandb.plot_table(
        "jz90/stepped_cm",
        wandb.Table(columns=["Step", "Actual", "Predicted", "nPredictions"], data=data),
        fields,
        {"title": title},
    )})

if __name__ == '__main__':
    wandb.init(project="dev", config={"training": {}})

    train_data = datasets.MNIST('../data', train=True, download=True, transform=ToTensor())

    loader = DataLoader(train_data, batch_size=512)

    t = Trainer(loader, loader, 10, no_cuda=True)
    t.batch = 0
    t.epoch = 0
    t.sample = 0
    confusion_matrix(t, "cm", y_true=[0, 0, 1, 1, 2, 2], preds=[0,1,1,2,2,0],
                                                              title="Val Conf. Mat.", panel="test")


    t.batch = 2
    t.epoch = 1
    t.sample = 4
    confusion_matrix(t, "cm", y_true=[0, 0, 1, 1, 2, 2], preds=[0,1,0,2,2,0],
                                                              title="Val Conf. Mat.", panel="test")

    t.batch = 3
    t.epoch = 2
    t.sample = 6
    confusion_matrix(t, "cm", y_true=[0, 0, 1, 1, 2, 2], preds=[0,1,0,1,1,0],
                                                              title="Val Conf. Mat.", panel="test")

    wandb.finish()