from unittest import TestCase

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from src.pthelpers.util.normilization import get_mean_and_std


class Test(TestCase):
    def test_get_mean_and_std(self):
        data = torchvision.datasets.MNIST("./data/", download=True, transform=ToTensor())

        dataloader = DataLoader(data, batch_size=1024, shuffle=True)

        mean, std = get_mean_and_std(dataloader)

        self.assertEqual(mean.item(), 0.13066695630550385)
        self.assertEqual(std.item(), 0.3081152141094208)
