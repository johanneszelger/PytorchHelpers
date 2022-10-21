import copy

import torch
import torch.nn.functional as F
import wandb
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, _LRScheduler
from tqdm import tqdm


def should_use_cuda(no_cuda):
    """
    Determins wether cuda should be used
    :param no_cuda: flag to force result to be False
    :return: if cuda should be used
    """
    use_cuda = not no_cuda and torch.cuda.is_available()
    if use_cuda:
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class Trainer:
    """
    Helper class that enables easy training, etc for pytorch. Logging to w&b
    """


    def __init__(self, train_dl: DataLoader, val_dl: DataLoader, test_dl: DataLoader = None,
                 loss_fn: nn.Module = None,
                 no_cuda: bool = False,
                 metrics: dict = None) -> None:
        """
        :param train_dl: train data
        :param val_dl: validation data
        :param test_dl: test data
        :param loss_fn: loss function, defaults to negative log likelihood
        :param no_cuda: flag to force cpu usage
        :param metrics: dictionary for all metrics to use
        """
        self.loss_fn = loss_fn if loss_fn is not None else F.nll_loss
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        self.metrics = metrics if metrics is not None else {}
        self.__val_metrics = copy.deepcopy(self.metrics)

        self.__logging_infos = {}
        self.device = should_use_cuda(no_cuda)
        self.best_validation_loss = float("inf")

        if "log_interval_batches" not in wandb.config:
            wandb.config["log_interval_batches"] = 100
        if "dry_run" not in wandb.config:
            wandb.config["dry_run"] = False
        if "save_every_nth_epoch" not in wandb.config and "cp_base_path" in wandb.config:
            wandb.config["save_every_nth_epoch"] = 1


    def __reset(self):
        self.epoch = 0
        self.batch = 0
        self.sample = 0


    def __to(self, device, model):
        model.to(device)
        if hasattr(self.loss_fn, 'to'):
            self.loss_fn.to(device)
        for metric in self.metrics.values():
            metric.to(device)
        for metric in self.__val_metrics.values():
            metric.to(device)


    def train(self, model: nn.Module, optimizer: Optimizer, epochs: int, scheduler: _LRScheduler = None) -> None:
        """
        Train a model
        :param model: the model to train
        :param optimizer: opt to use
        :param epochs: how many epochs to train
        :param scheduler: scheduler to use
        :return: None
        """
        # prepare training
        self.__reset()
        self.__to(self.device, model)
        start_epoch = 1
        self.__logging_infos["end_epoch"] = epochs
        self.__logging_infos["running_loss"] = 0

        # train for x epochs
        for self.epoch in range(start_epoch, epochs + 1):
            self.__train_epoch(model, optimizer)
            self.test(model, self.val_dl)
            if scheduler is not None:
                scheduler.step()
            self.__epoch_end_log()
            if "save_every_nth_epoch" in wandb.config and self.epoch % wandb.config["save_every_nth_epoch"] == 0:
                from src.training.persist import save_training_state
                save_training_state(self, model, optimizer)

        # training is done, test the model
        if self.test_dl:
            self.test(model, self.test_dl)


    def __train_epoch(self, model: nn.Module, optimizer: Optimizer):
        model.train()

        with tqdm(self.train_dl, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {self.epoch}/{self.__logging_infos['end_epoch']}")
            for batch_idx, (data, target) in enumerate(tepoch, 0):
                self.batch += 1
                self.sample += len(data)

                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = self.loss_fn(output, target)
                loss.backward()
                optimizer.step()

                self.__logging_infos["running_loss"] += loss.item()
                for name, metric in self.metrics.items():
                    metric.update(output, target.int())

                tepoch.set_postfix(loss=loss.item())

                if self.__inter_epoch_log() and wandb.config["dry_run"]:
                    break


    def test(self, model: nn.Module, test_loader: DataLoader) -> None:
        """
        Tests a model
        :param model: model to test
        :param test_loader: data to use for testing
        :return: None
        """

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader) * test_loader.batch_size,
                100. * correct / len(test_loader) / test_loader.batch_size))


    def __inter_epoch_log(self) -> bool:
        if self.batch % wandb.config["log_interval_batches"] == 0:
            self.__log()
            return True
        return False


    def __epoch_end_log(self) -> bool:
        if self.batch % wandb.config["log_interval_batches"] is None:
            self.__log()
            return True
        return False


    def __log(self) -> None:
        print('\nTrain Epoch: {} [{}/{} ({:.0f}%)]\tAvg loss: {:.6f}\n'.format(
                self.epoch, self.sample, len(self.train_dl),
                100. * self.batch / len(self.train_dl), self.__logging_infos["running_loss"] / wandb.config["log_interval_batches"]))
        self.__logging_infos["running_loss"] = 0


