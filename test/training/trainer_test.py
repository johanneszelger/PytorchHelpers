import os
import shutil
import unittest
import time
from unittest.mock import MagicMock

import torch
import wandb
from torch import optim
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from models.simple_net import SimpleNet
from src.training.trainer import Trainer


class Test(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName)
        wandb.init(mode="disabled")
        wandb.config.update({"cp_base_path": "checkpoints"})

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

        self.train_loader = DataLoader(train_data, batch_size=500)
        self.test_loader = DataLoader(test_data, batch_size=500)
        self.model = SimpleNet()
        self.optimizer = Adam(self.model.parameters())
    def tearDown(self) -> None:
        time.sleep(0.5)
        shutil.rmtree(wandb.config["cp_base_path"], ignore_errors=True)

    def test_train(self):
        wandb.run.name = "test_train"
        wandb.config.update({"val_interval_batches": 1})
        trainer = Trainer(self.train_loader, self.test_loader, self.test_loader)

        trainer._Trainer__train_epoch = MagicMock(return_value=None)
        train_epoch = trainer._Trainer__train_epoch
        trainer.test = MagicMock(return_value=None)

        epochs = 5
        trainer.train(self.model, self.optimizer, epochs)
        assert train_epoch.call_count == epochs, f"Expected {epochs} calls for train_epoch, got {train_epoch.call_count}"
        assert trainer.test.call_count == 1, f"Expected {1} calls for test, got {trainer.test.call_count}"

        cp_dir = os.path.join("checkpoints", "test_train")
        files = os.listdir(cp_dir)
        assert len(files) == epochs, f"Expected {epochs} epoch checkpoints, got {len(files)}"

if __name__ == '__main__':
    unittest.main()
