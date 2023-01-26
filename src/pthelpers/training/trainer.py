import copy
import logging
import time
from typing import Union, Tuple

import torch
import torch.nn.functional as F
import wandb
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from pthelpers.plotting.class_dist import plot_class_dist
from pthelpers.utils.class_names import get_class_names
from pthelpers.utils.reproducibility import get_seed


def should_use_cuda(no_cuda):
    """
    Determines whether cuda should be used
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
                 metrics: dict = None,
                 sampler: Sampler = None) -> None:
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
        self.sampler = sampler
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
        if "cp_base_path" not in self.config:
            self.config["cp_base_path"] = None
        if "save_every_nth_epoch" not in self.config and "cp_base_path" in self.config:
            self.config["save_every_nth_epoch"] = 1
        if "cleanup_after_training" not in self.config:
            self.config["cleanup_after_training"] = True

        if "unfreeze_after" not in self.config:
            self.config["unfreeze_after"] = None

        if "plot" not in self.config:
            self.config["plot"] = True
        if "plot_class_dist" not in self.config:
            self.config["plot_class_dist"] = False
        if "plot_samples_training_start" not in self.config:
            self.config["plot_samples_training_start"] = True
        if "plot_samples_training_log" not in self.config:
            self.config["plot_samples_training_log"] = True
        if "plot_samples_validation_log" not in self.config:
            self.config["plot_samples_validation_log"] = True
        if "plot_confusion_training_log" not in self.config:
            self.config["plot_confusion_training_log"] = True
        if "plot_confusion_validation_log" not in self.config:
            self.config["plot_confusion_validation_log"] = True

    def __reset(self):
        self.epoch = 0
        self.batch = 0
        self.sample = 0

        self.collected_targets = Tensor([]).detach()
        self.collected_outputs = Tensor([]).detach()

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
        if self.config["plot_samples_training_start"]:
            self.plot_data(self.train_dl, "training start")

        # prepare training
        self.__reset()
        self.__to(self.device, model)
        from pthelpers.training.persist import load_latest
        start_epoch = load_latest(self, model, optimizer)
        self.__logging_infos["end_epoch"] = epochs
        self.__logging_infos["running_loss"] = 0

        # train for x epochs
        for self.epoch in range(start_epoch, epochs + 1):
            if self.sampler:
                self.sampler.set_epoch(self.epoch)
            resume = self.__train_epoch(model, optimizer)
            if scheduler is not None:
                scheduler.step()

            if not resume:
                break

        # training is done, test the model
        # if self.test_dl:
        #     self.test(model, self.test_dl)

        from pthelpers.training.persist import clean_checkpoints
        clean_checkpoints()

    def __train_epoch(self, model: nn.Module, optimizer: Optimizer):
        if self.config["unfreeze_after"] is not None and self.epoch == self.config["unfreeze_after"]:
            print("unfreezing model")
            self.__unfreeze_model__(model)

        model.train()

        with tqdm(self.train_dl, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {self.epoch}/{self.__logging_infos['end_epoch']}")
            for batch_idx, samples in enumerate(tepoch, 0):
                data = samples[0]
                target = samples[1]
                self.collected_targets = torch.cat((self.collected_targets, target.detach())) if target.ndim == 1 else \
                    torch.cat((self.collected_targets, target.detach().argmax(axis=-1)))
                paths = (samples + [None])[2]

                self.batch += 1
                self.sample += len(data)

                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                self.collected_outputs = torch.cat((self.collected_outputs, output.detach().cpu()))
                loss = self.loss_fn(output, target)
                loss.backward()
                optimizer.step()

                self.__logging_infos["running_loss"] += loss.item()
                for name, metric in self.metrics.items():
                    metric.update(output, target.int())

                tepoch.set_postfix(loss=loss.item())

                if self.__inter_epoch_training_log(optimizer, model) and self.config["dry_run"]:
                    return False

                self.__inter_epoch_validation(model, optimizer)

        end_of_epoch_logged = self.__epoch_end_training_log(optimizer, model)
        self.__epoch_end_validation(model, optimizer)

        if "save_every_nth_epoch" in self.config and self.epoch % self.config["save_every_nth_epoch"] == 0:
            from pthelpers.training.persist import save_training_state
            save_training_state(self, model, optimizer)

        if end_of_epoch_logged and self.config["dry_run"]:
            return False

        return True

    def test(self, model: nn.Module, test_loader: DataLoader, metrics: Union[dict, None] = None) -> Tuple[
        int, Tensor, Tensor]:
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
            targets = Tensor([])
            outputs = Tensor([])
            for samples in test_loader:
                data = samples[0]
                target = samples[1]
                targets = torch.cat((targets, target)) if target.ndim == 1 else torch.cat(
                    (targets, target.argmax(axis=-1)))
                paths = (samples + [None])[2]

                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                outputs = torch.cat((outputs, output.cpu()))
                test_loss += self.loss_fn(output, target).item()

                for name, metric in metrics.items():
                    metric.update(output, target.int())

        test_loss /= len(test_loader)

        model.train()
        return test_loss, targets, outputs

    def __inter_epoch_training_log(self, optimzer: Optimizer, model: nn.Module) -> bool:
        if self.config["log_interval_batches"] is not None \
                and self.batch % self.config["log_interval_batches"] == 0:
            self.__training_log(optimzer, model)
            return True
        return False

    def __epoch_end_training_log(self, optimzer: Optimizer, model: nn.Module) -> bool:
        if self.config["log_interval_batches"] is None:
            self.__training_log(optimzer, model)
            return True
        return False

    def __training_log(self, optimizer: Optimizer, model: nn.Module) -> None:
        batch_in_epoch = self.batch - (len(self.train_dl) * (self.epoch - 1))
        batches_since_last_log = self.config["log_interval_batches"] if self.config["log_interval_batches"] is not None \
            else len(self.train_dl)
        if self.config["print_logs"]:
            print('\nTrain Epoch: {} [{}/{} ({:.0f}%)]\tAvg loss: {:.6f}\n'.format(
                self.epoch, batch_in_epoch, len(self.train_dl),
                100. * batch_in_epoch / len(self.train_dl),
                self.__logging_infos["running_loss"] / batches_since_last_log))

        data = {"t_loss": self.__logging_infos["running_loss"] / batches_since_last_log,
                "lr": optimizer.param_groups[0]['lr']}

        for name, metric in self.metrics.items():
            res = metric.compute()
            data[name] = res.item()
            metric.reset()

        self.wandb_log(data, "training results/")

        self.__logging_infos["running_loss"] = 0

        if self.config["plot_samples_training_log"]:
            self.plot_data(self.train_dl, "training", model)

        if self.config["plot"] and self.config["plot_confusion_training_log"]:
            from pthelpers.logging.confusion_matrix import confusion_matrix
            confusion_matrix(self, "training_cm", self.collected_targets.numpy(),
                             self.collected_outputs.numpy().argmax(axis=-1),
                             get_class_names(self.n_classes), title="Training CM", panel="training results")
            # wandb.log({"training_cm" : wandb.plot.confusion_matrix(probs=None,
            #                                                 y_true=self.collected_targets.numpy(), preds= self.collected_outputs.numpy().argmax(axis=-1),
            #                                                 class_names=get_class_names(self.n_classes))})

        self.collected_targets = Tensor([]).detach()
        self.collected_outputs = Tensor([]).detach()

    def wandb_log(self, data: dict, add_prefix: str = None):
        prefixed_data = {}
        if add_prefix is not None:
            for k, v in data.items():
                if "/" not in k:
                    prefixed_data[add_prefix + k] = v
                else:
                    prefixed_data[k] = v
        else:
            prefixed_data = data

        prefixed_data["Hidden Panels/epoch"] = float(self.batch) / len(self.train_dl)
        prefixed_data["Hidden Panels/batch"] = self.batch
        prefixed_data["Hidden Panels/sample"] = self.sample

        wandb.log(prefixed_data, step=self.batch)

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
        start = time.time()
        loss, targets, outputs = self.test(model, self.val_dl, self.__val_metrics)
        logging.debug(f"validation testing took {time.time() - start}")

        batch_in_epoch = self.batch - (len(self.train_dl) * (self.epoch - 1))
        if self.config["print_logs"]:
            print('\nValidation Epoch: {} [{}/{} ({:.0f}%)]\tAvg loss: {:.6f}\n'.format(
                self.epoch, batch_in_epoch, len(self.train_dl),
                100. * batch_in_epoch / len(self.train_dl), loss))

        data = {"v_loss": loss}
        for name, metric in self.__val_metrics.items():
            res = metric.compute()
            data["v_" + name] = res.item()
            metric.reset()

        self.wandb_log(data, "validation results/")

        if loss < self.best_validation_loss:
            from pthelpers.training.persist import save_training_state
            save_training_state(self, model, optimizer, "best.pth")

        if self.config["plot_samples_validation_log"]:
            self.plot_data(self.val_dl, "validation", model)

        if self.config["plot"] and self.config["plot_confusion_validation_log"]:
            from pthelpers.logging.confusion_matrix import confusion_matrix
            confusion_matrix(self, "validation_cm", targets.numpy(), outputs.numpy().argmax(axis=-1),
                             get_class_names(self.n_classes), title="Validation CM", panel="validation results")

    def __unfreeze_model__(self, model: nn.Module):
        for param in model.parameters():
            param.requires_grad = True

    def plot_class_dist(self):
        if self.config["plot"] and self.config["plot_class_dist"]:
            print("plotting class dist, depending on dataset this might take some time")
            plot_class_dist(self.train_dl, self.n_classes)

    def plot_data(self, dl: DataLoader, name: str, model: nn.Module = None):
        from pthelpers.plotting.samples import plot_samples, plot_samples_with_predictions
        if self.config["plot"]:
            if (model is None):
                plot_samples(dl, self.n_classes, name, panel=f"{name} label plots")
            else:
                plot_samples_with_predictions(self, dl, self.n_classes, name, model, panel=f"{name} prediction plots")
