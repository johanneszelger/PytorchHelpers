import os

import pandas as pd
import torch.nn.functional
import wandb
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.data.dataset import T_co

from mnist_test import MnistTest
from pthelpers.plotting.class_dist import plot_class_dist


class PlotClassDistTest(MnistTest):
    def setUp(self) -> None:
        super(PlotClassDistTest, self).setUp()


    def test_plot_dataloader(self) -> None:
        plot_class_dist(self.train_loader, 10)


    class Data2d(Dataset):
        def __init__(self, dl):
            self.dl = dl
            self.transform = None


        def __getitem__(self, index) -> T_co:
            x, y = self.dl.dataset.__getitem__(index)
            y = torch.nn.functional.one_hot(torch.Tensor([y]).to(torch.int64), 10)
            return x, y

        def __len__(self):
            return len(self.dl.dataset)


    def test_plot_dataloader_2d(self) -> None:
        dl = DataLoader(self.Data2d(self.train_loader), 500)
        plot_class_dist(dl, 10)


    def test_samples(self) -> None:
        data = self.train_loader.dataset
        targets = pd.Series(data.targets)

        weights = 1/targets.value_counts().to_numpy()
        sample_weights = [weights[i] for i in targets]
        sampler = WeightedRandomSampler(sample_weights, replacement=True, num_samples=len(data) * 10)
        dl = DataLoader(data, batch_size=500, sampler=sampler)

        plot_class_dist(dl, 10)
