import os
import shutil
import unittest

import torch
import wandb
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from models.simple_net import SimpleNet
from src.training.trainer import Trainer


class TestSum(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName)
        wandb.init(mode="disabled")
        wandb.config.update({"cp_base_path": "checkpoints", "log_interval_batches": 5, "save_every_nth_epoch": 8})

    def test_mnist_training(self):
        torch.manual_seed(42)

        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        train_data = datasets.MNIST('../data', train=True, download=True,
                           transform=transform)
        train_data.data = train_data.data[:1000]
        test_data = datasets.MNIST('../data', train=False,
                           transform=transform)
        test_data.data = test_data.data[:1000]

        train_loader = DataLoader(train_data, batch_size=500)
        test_loader = DataLoader(test_data, batch_size=500)

        model = SimpleNet()
        optimizer = optim.Adadelta(model.parameters(), lr=0.1)

        scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

        Trainer(train_loader, test_loader).train(model, optimizer, 10, scheduler)

        cp_dir = os.path.join("checkpoints", wandb.run.name)
        files = os.listdir(cp_dir)
        shutil.rmtree(cp_dir)
        assert len(files) == 2, "Expected best and one epoch cp"

if __name__ == '__main__':
    unittest.main()