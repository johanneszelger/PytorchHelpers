import unittest

import torch
import wandb
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms, datasets

from models.simple_net import SimpleNet
from src.training.trainer import Trainer


class TestSum(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName)
        wandb.init(mode="disabled")
        wandb.config.update({"cp_base_path": "checkpoints"})

    def test_mnist_training(self):
        torch.manual_seed(42)

        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        train_data = datasets.MNIST('../data', train=True, download=True,
                           transform=transform)
        test_data = datasets.MNIST('../data', train=False,
                           transform=transform)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=512)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=512)

        model = SimpleNet()
        optimizer = optim.Adadelta(model.parameters(), lr=0.1)

        scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

        Trainer(train_loader, test_loader).train(model, optimizer, 10, scheduler)

if __name__ == '__main__':
    unittest.main()