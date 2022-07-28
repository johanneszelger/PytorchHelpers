from unittest import TestCase

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from src.pthelpers.utils.normilization import get_mean_and_std


class Test(TestCase):
    def test_get_mean_and_std(self):
        data = torchvision.datasets.MNIST("./data/", download=True, transform=ToTensor())

        dataloader = DataLoader(data, batch_size=1024, shuffle=True)

        mean, std = get_mean_and_std(dataloader)

        self.assertAlmostEqual(mean.item(), 0.1307, 3)
        self.assertAlmostEqual(std.item(), 0.3081, 3)
