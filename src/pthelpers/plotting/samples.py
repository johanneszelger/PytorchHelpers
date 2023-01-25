import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets
from torchvision.transforms import ToTensor

from pthelpers.models import SimpleNet
from pthelpers.training import Trainer
from pthelpers.utils.class_names import get_class_names


def plot_samples(dl: DataLoader, n_classes: int, data_name: str = "training"):
    dataset = dl.dataset
    n_cols = wandb.config["samples_per_row"] if "samples_per_row" in wandb.config else 5
    tbl = wandb.Table(columns=["cls"] + ["sample_" + str(i + 1) for i in range(n_cols)])

    cls_names = get_class_names(n_classes)

    g = torch.Generator()
    g.manual_seed(42)
    for i in range(n_classes):
        weights = np.zeros(n_classes)
        if dataset.targets.ndim == 1 or dataset.targets.shape[1] == 1:
            weights[i] = 1
            sample_weights = [weights[j] for j in dataset.targets]
        else:
            weights[i] = dataset.targets[:, i].sum()
        sampler = WeightedRandomSampler(sample_weights, 10000, generator=g)
        loader = DataLoader(dataset, batch_size=n_cols, sampler=sampler)

        for data in loader:
            imgs = data[0]
            tbl.add_data(cls_names[i], *[wandb.Image(img) for img in imgs])
            break

    wandb.log({f"data/sample {data_name} images": tbl})


def plot_samples_with_predictions(trainer: Trainer, dl: DataLoader, n_classes: int, data_name: str, model: nn.Module, batch_size=32,
                                  panel: str = None):
    model.eval()
    with torch.no_grad():
        dataset = dl.dataset
        n_samples = wandb.config["pred_plot_samples_per_class"] if "pred_plot_samples_per_class" in wandb.config else 50
        tbl = wandb.Table(columns=["image", "label", "pred"])

        cls_names = get_class_names(n_classes)

        g = torch.Generator()
        g.manual_seed(42)
        for i in range(n_classes):
            weights = np.zeros(n_classes)
            weights[i] = 1
            sample_weights = [weights[i] for i in dataset.targets]
            sampler = WeightedRandomSampler(sample_weights, 10000, generator=g)
            loader = DataLoader(dataset, batch_size=min(batch_size, n_samples), sampler=sampler)

            for j, data in enumerate(loader):
                sub_batch_size = min(batch_size, n_samples - j * batch_size)
                imgs = data[0][:sub_batch_size].to(trainer.device)
                targets = data[1][:sub_batch_size]
                preds = model(imgs).cpu().numpy().argmax(axis=1)
                [tbl.add_data(wandb.Image(imgs[i]), cls_names[targets[i].argmax()], cls_names[preds[i]]) for i in range(len(imgs))]
                if n_samples - (j + 1) * batch_size <= 0:
                    break

    trainer.wandb_log({f"{panel + '/' if panel is not None else ''}{data_name} predictions": tbl})
    model.train()


if __name__ == '__main__':
    wandb.init(project="dev", config={"training": {}})

    train_data = datasets.MNIST('../data', train=True, download=True, transform=ToTensor())

    loader = DataLoader(train_data, batch_size=512)

    plot_samples(loader, 10, data_name="test2")

    t = Trainer(loader, loader, 10, no_cuda=True)
    t.batch = 0
    t.epoch = 0
    t.sample = 0
    plot_samples_with_predictions(t, loader, 10, model=SimpleNet(), data_name="test2")


    t.batch = 2
    t.epoch = 1
    t.sample = 4
    plot_samples_with_predictions(t, loader, 10, model=SimpleNet(), data_name="test2")
