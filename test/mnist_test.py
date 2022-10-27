import shutil
import time
import unittest
from typing import Tuple

import torch
import wandb
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

from pthelpers.models import SimpleNet


class MnistTest(unittest.TestCase):
    def __init__(self, methodName, size: Tuple = None, rgb: bool = False, batch_size=500):
        super().__init__(methodName)

        transformations = [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]

        if size is not None:
            transformations.append(transforms.Resize((size[0], size[1])))

        if rgb:
            transformations.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x))

        transform = transforms.Compose(transformations)
        train_data = datasets.MNIST('../data', train=True, download=True,
                                    transform=transform)
        train_data.data = train_data.data[:2*batch_size]
        test_data = datasets.MNIST('../data', train=False,
                                   transform=transform)
        test_data.data = test_data.data[:2*batch_size]

        self.train_loader = DataLoader(train_data, batch_size=batch_size)
        self.test_loader = DataLoader(test_data, batch_size=batch_size)
        self.model = SimpleNet()
        self.optimizer = Adam(self.model.parameters())
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.9)


    def setUp(self) -> None:
        wandb.init(mode="disabled")
        wandb.config.update({"training": {"cp_base_path": "checkpoints"}})
        torch.manual_seed(42)


    def tearDown(self) -> None:
        time.sleep(0.5)
        shutil.rmtree(wandb.config["training"]["cp_base_path"], ignore_errors=True)