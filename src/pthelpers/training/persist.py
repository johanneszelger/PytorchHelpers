import os
import os.path as osp

import dill
import torch
import wandb
from torch import nn
from torch.nn import Sequential
from torch.optim import Optimizer

from pthelpers.training.trainer import Trainer


def generate_run_path():
    if not wandb.config["training"]["cp_base_path"]:
        return None

    path = osp.join(wandb.config["training"]["cp_base_path"], wandb.run.name)

    os.makedirs(path, exist_ok=True)

    return path

def load_training_state(trainer: Trainer, model: nn.Module, optimizer: Optimizer, name: str) -> None:
    """
    loads a training state from a file
    :param trainer: Trainer to load
    :param model: model params to load
    :param optimizer: optimizer params to load
    :return: None
    """
    checkpoint = torch.load(osp.join(generate_run_path(), name))
    trainer.epoch = checkpoint['epoch']
    trainer.batch = checkpoint['batch']
    trainer.sample = checkpoint['sample']
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
    if path is None:
        return

    print(f'\nSaving checkpoint: {path}\n')

    if name is None:
        name = f"epoch_{trainer.epoch}.pth"

    torch.save({'epoch': trainer.epoch,
                'batch': trainer.batch,
                'sample': trainer.sample,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_loss': trainer.best_validation_loss,
                },
               osp.join(path, name), pickle_module=dill)


def clean_checkpoints():
    cp_dir = generate_run_path()
    if wandb.config["training"]["cleanup_after_training"] and cp_dir:
        files = os.listdir(cp_dir)
        last = max([int(f[6:f.index(".")]) for f in files if not f.startswith("best")])
        for f in files:
            if f == f'epoch_{last}.pth':
                os.rename(osp.join(cp_dir, f'epoch_{last}.pth'), osp.join(cp_dir, 'final.pth'))
            elif f == 'best.pth':
                continue
            else:
                os.remove(osp.join(cp_dir, f))


def load_latest(trainer: Trainer, model: nn.Module, optimizer: Optimizer) -> int:
    cp_dir = generate_run_path()
    if wandb.config["training"]["warm_start"] and cp_dir:
        files = os.listdir(cp_dir)
        epochs = [int(f[6:f.index(".")]) for f in files if not f.startswith("best")]
        if len(epochs) > 0:
            load_training_state(trainer, model, optimizer, f"epoch_{max(epochs)}.pth")
            return max(epochs) + 1

    return 1
