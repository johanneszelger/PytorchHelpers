import copy
import logging
from typing import Union

import torch
import torch.nn.functional as F
import wandb
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from pthelpers.plotting.class_dist import plot_class_dist
from pthelpers.plotting.samples import plot_samples
from pthelpers.utils.reproducibility import get_seed


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


    def __init__(self, train_dl: DataLoader, val_dl: DataLoader, n_classes: int, test_dl: DataLoader = None,
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
        self.n_classes = n_classes
        self.metrics = metrics if metrics is not None else {}
        self.__val_metrics = copy.deepcopy(self.metrics)

        self.__logging_infos = {}
        self.device = should_use_cuda(no_cuda)
        self.best_validation_loss = float("inf")

        self.config = wandb.config["training"]

        if "log_interval_batches" not in self.config:
            self.config["log_interval_batches"] = 100
        if "val_interval_batches" not in self.config:
            self.config["val_interval_batches"] = None
        if "print_logs" not in self.config:
            self.config["print_logs"] = True
        if "dry_run" not in self.config:
            self.config["dry_run"] = False
        if "warm_start" not in self.config:
            self.config["warm_start"] = True
        if "cleanup_after_training" not in self.config:
            self.config["cleanup_after_training"] = True
        if "cp_base_path" not in self.config:
            self.config["cp_base_path"] = None
        if "save_every_nth_epoch" not in self.config and "cp_base_path" in self.config:
            self.config["save_every_nth_epoch"] = 1

        if "plot_class_dist" not in self.config:
            self.config["plot_class_dist"] = True
        if "log_interval_batches" not in self.config:
            self.config["plot_data_aug"] = True


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
        if get_seed() is None:
            print("No seed set, results might not be reproducible!")

        self.plot_class_dist()
        self.plot_data_aug()

        # prepare training
        self.__reset()
        self.__to(self.device, model)
        from pthelpers.training.persist import load_latest
        start_epoch = load_latest(self, model, optimizer)
        self.__logging_infos["end_epoch"] = epochs
        self.__logging_infos["running_loss"] = 0

        # train for x epochs
        for self.epoch in range(start_epoch, epochs + 1):
            result = self.__train_epoch(model, optimizer)
            if scheduler is not None:
                scheduler.step()

            end_of_epoch_logged = self.__epoch_end_training_log(optimizer)
            self.__epoch_end_validation(model, optimizer)

            if "save_every_nth_epoch" in self.config and self.epoch % self.config["save_every_nth_epoch"] == 0:
                from pthelpers.training.persist import save_training_state
                save_training_state(self, model, optimizer)

            if not result:
                break
            if end_of_epoch_logged and self.config["dry_run"]:
                break

        # training is done, test the model
        # if self.test_dl:
        #     self.test(model, self.test_dl)

        from pthelpers.training.persist import clean_checkpoints
        clean_checkpoints()


    def __train_epoch(self, model: nn.Module, optimizer: Optimizer):
        model.train()

        with tqdm(self.train_dl, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {self.epoch}/{self.__logging_infos['end_epoch']}")
            for batch_idx, samples in enumerate(tepoch, 0):
                data = samples[0]
                target = samples[1]
                paths = (samples + [None])[2]

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

                if self.__inter_epoch_training_log(optimizer) and self.config["dry_run"]:
                    return False

                self.__inter_epoch_validation(model, optimizer)

        return True


    def test(self, model: nn.Module, test_loader: DataLoader, metrics: Union[dict, None] = None) -> float:
        """
        Tests a model
        :param model: model to test
        :param test_loader: data to use for testing
        :return: None
        """

        if metrics is None:
            metrics = {}

        for name, metric in metrics.items():
            metric.reset()

        model.eval()
        test_loss = 0

        with torch.no_grad():
            for samples in test_loader:
                data = samples[0]
                target = samples[1]
                paths = (samples + [None])[2]

                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += self.loss_fn(output, target).item()

                for name, metric in metrics.items():
                    metric.update(output, target.int())

        test_loss /= len(test_loader)

        return test_loss


    def __inter_epoch_training_log(self, optimzer: Optimizer) -> bool:
        if self.config["log_interval_batches"] is not None \
                and self.batch % self.config["log_interval_batches"] == 0:
            self.__training_log(optimzer)
            return True
        return False


    def __epoch_end_training_log(self, optimzer: Optimizer) -> bool:
        if self.config["log_interval_batches"] is None:
            self.__training_log(optimzer)
            return True
        return False


    def __training_log(self, optimizer: Optimizer) -> None:
        batch_in_epoch = self.batch - (len(self.train_dl) * (self.epoch - 1))
        batches_since_last_log = self.config["log_interval_batches"] if self.config["log_interval_batches"] is not None \
            else len(self.train_dl)
        if self.config["print_logs"]:
            print('\nTrain Epoch: {} [{}/{} ({:.0f}%)]\tAvg loss: {:.6f}\n'.format(
                    self.epoch, batch_in_epoch, len(self.train_dl),
                    100. * batch_in_epoch / len(self.train_dl),
                    self.__logging_infos["running_loss"] / batches_since_last_log))

        data = {"t_loss": self.__logging_infos["running_loss"] / batches_since_last_log, "lr": optimizer.param_groups[0]['lr']}

        for name, metric in self.metrics.items():
            data[name] = metric.compute().item()
            metric.reset()

        self.__wandb_log(data)

        self.__logging_infos["running_loss"] = 0


    def __wandb_log(self, data: dict):
        data["epoch"] = self.batch / len(self.train_dl)
        data["batch"] = self.batch
        data["sample"] = self.sample
        wandb.log(data)


    def __inter_epoch_validation(self, model: nn.Module, optimizer: Optimizer) -> bool:
        if self.config["val_interval_batches"] is not None \
                and self.batch % self.config["val_interval_batches"] == 0:
            self.__validate(model, optimizer)
            return True
        return False


    def __epoch_end_validation(self, model: nn.Module, optimizer: Optimizer) -> bool:
        if self.config["val_interval_batches"] is None:
            self.__validate(model, optimizer)
            return True
        return False


    def __validate(self, model: nn.Module, optimizer: Optimizer) -> None:
        loss = self.test(model, self.val_dl, self.__val_metrics)

        batch_in_epoch = self.batch - (len(self.train_dl) * (self.epoch - 1))
        if self.config["print_logs"]:
            print('\nValidation Epoch: {} [{}/{} ({:.0f}%)]\tAvg loss: {:.6f}\n'.format(
                    self.epoch, batch_in_epoch, len(self.train_dl),
                    100. * batch_in_epoch / len(self.train_dl), loss))

        data = {"v_loss": loss}
        for name, metric in self.__val_metrics.items():
            data["v_" + name] = metric.compute().item()

        self.__wandb_log(data)

        if loss < self.best_validation_loss:
            from pthelpers.training.persist import save_training_state
            save_training_state(self, model, optimizer, "best.pth")


    def plot_class_dist(self):
        if self.config["plot_class_dist"]:
            print("plotting class dist, depending on dataset this might take some time")
            plot_class_dist(self.train_dl, self.n_classes)


    def plot_data_aug(self):
        plot_samples(self.train_dl, self.n_classes, )



