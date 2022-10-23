import unittest

import torch
import wandb
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms, datasets

from models.simple_net import SimpleNet
from src.training.trainer import Trainer


class Test(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName)
        wandb.init(mode="disabled")
        wandb.config.update({"cp_base_path": "checkpoints"})

    def test_train(self):
        train_data = datasets.MNIST('../data', train=True, download=True)
        test_data = datasets.MNIST('../data', train=False)
        trainer = Trainer(train_data, test_data)

        trainer.train()
if __name__ == '__main__':
    unittest.main()