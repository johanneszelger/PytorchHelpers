import os

import dill
import torch
import wandb
import os.path as osp

from torch import nn
from torch.nn import Sequential
from torch.optim import Optimizer

from src.training.trainer import Trainer


def load_training_state(trainer: Trainer, model: nn.Module, optimizer: Optimizer) -> None:
    """
    loads a training state from a file
    :param trainer: Trainer to load
    :param model: model params to load
    :param optimizer: optimizer params to load
    :return: None
    """
    checkpoint = torch.load(generate_run_path())
    trainer.epoch = checkpoint['epoch'] + 1
    trainer.best_validation_loss = checkpoint['best_loss']

    optimizer.load_state_dict(checkpoint['optimizer'])

    if isinstance(model, Sequential):
        model[0].load_state_dict({k[2:]: v for (k,v) in checkpoint['state_dict'].items()})
    else:
        model.load_state_dict(checkpoint['state_dict'])


def save_training_state(trainer: Trainer, model: nn.Module, optimizer: Optimizer, name: str= None) -> None:
    """
    saves a training state from a file
    :param trainer: Trainer to save
    :param model: model params to save
    :param optimizer: optimizer params to save
    :return: None
    """
    path = generate_run_path()
    print(f'\nSaving checkpoint: {path}\n')

    if name is None:
        name = f"epoch_{trainer.epoch}.pth"

    torch.save({'epoch': trainer.epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_loss': trainer.best_validation_loss,
                },
               osp.join(path, name), pickle_module=dill)


def generate_run_path():
    if not wandb.config["cp_base_path"]:
        raise ValueError("configure cp_base_path to safe models")

    path = osp.join(wandb.config["cp_base_path"], wandb.run.name)

    os.makedirs(path, exist_ok=True)

    return path