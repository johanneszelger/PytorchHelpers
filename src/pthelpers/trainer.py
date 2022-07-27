""" Providing Trainer that automates pthelpers training """
import copy
import os
import os.path as osp

import dill
import torch
from sacred import Ingredient
from sacred.run import Run
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from tqdm import tqdm

try:
    from pthelpers.reproducibility import Reproducer
except:
    from src.pthelpers.reproducibility import Reproducer

trainer_ingredient = Ingredient('trainer')


@trainer_ingredient.config
def cfg():
    cp_dir = None
    cp_dir_append_experiment = False
    cp_dir_append_run = True
    remove_cp_after_training = True
    epochs = 1
    unfreeze_after = None
    use_gpu = True
    log_every_n_batches = None
    val_every_n_batches = None
    ignore_reproducibility = False


class Trainer:
    """
    Trainer that automates the training of models including features like automated checkpoint saving
    and logging
    """


    @trainer_ingredient.capture()
    def __load_existing_cp__(self, _log) -> None:
        cp_dir = self.__get_final_cp_dir__()
        if not cp_dir:
            return None

        epochs = [x.replace("checkpoint_", "").replace(".pth", "") for x in os.listdir(cp_dir) if
                  not x.startswith("best") and not osp.isdir(x)]
        if len(epochs) == 0:
            return None
        last_epoch = max(epochs)

        _log.info(f'Found existing checkpoint: checkpoint_{last_epoch}.pth in {cp_dir}')
        self.load(osp.join(cp_dir, f"checkpoint_{last_epoch}.pth"))


    @trainer_ingredient.capture()
    def __init__(self,
                 model: torch.nn.Module = None,
                 train_dataloader: DataLoader = None,
                 validation_dataloader: DataLoader = None,
                 loss_fn: torch.nn.Module = None,
                 optimizer: Optimizer = None,
                 metrics: dict = None,
                 # checkpoints_dir: str = None,
                 # log_dir: str = None,
                 # validate_every_steps: int = 100,
                 # remove_cp_after_training: bool = True,
                 # load_latest_existing=False,
                 # eval_first: bool = False,
                 # class_names: [] = None,
                 # use_gpu: bool = True \
                 _config=None) -> None:
        """
        :param model: the model to train
        :param train_dataloader: the data to train with
        :param loss_fn: the loss function for training
        :param optimizer: the optimizer for training
        :param metrics: the metrics apart from loss that should be printed and logged if log_dir is specified
        # :param checkpoints_dir: will produce a checkpoint for each epoch if specified and a "best" checkpoint
        #     if validation_dataloader is provided that is overwritten whenever there is a new minimal validation loss
        # :param log_dir: directory to log to (tensorboard format)
        # :param validation_dataloader: data to validate on
        # :param validate_every_steps: default every epoch gets validated, here steps can be defined to validate
        #     more often
        # :param remove_cp_after_training:
        #     if true removes all checkpoints except last and best after training
        # :param load_latest_existing:
        #     if true checks the checkpoint dir for the latest checkpoint and loads it and continues to train from there
        # :param eval_first:
        #     if true performs evaluation on test and valuation set before starting the training (can show anomalies)
        # :param class_names:
        #     list of classnames to use for metrics
        # :param use_gpu:
        #     if gpu acceleration shall be used
        """

        if model is None:
            raise ValueError("Model must be defined")
        if train_dataloader is None:
            raise ValueError("Train DL must be defined")
        if loss_fn is None:
            raise ValueError("Loss must be defined")
        if optimizer is None:
            raise ValueError("Optimizer must be defined")

        if metrics is None:
            metrics = {'accuracy': Accuracy()}

        self.model = model
        self.__train_dataloader = train_dataloader
        self.dl_len = len(self.__train_dataloader)
        self.__validation_dataloader = validation_dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metrics = metrics
        self.__val_metrics = copy.deepcopy(metrics)

        self.__best_validation_loss = None
        self.__epoch = 0

        self.batch_size = self.__train_dataloader.batch_size

        self.results = {}


    @trainer_ingredient.capture()
    def train(self, _log, _run: Run, _config) -> dict:
        """
        starts the training of the model
        :param _run: sacred experiment run
        :param _config: sacred experiment config
        :return: None
        """

        # send all to device here
        device = "cuda" if torch.cuda.is_available() and _config["use_gpu"] else "cpu"
        self.model.to(device)
        self.loss_fn.to(device)
        for metric in self.metrics.values():
            metric.to(device)
        for metric in self.__val_metrics.values():
            metric.to(device)


        # load existing here
        self.__load_existing_cp__()
        self.model.train()

        _log.info(f'Starting training')
        if not Reproducer.seed_set and not _config["ignore_reproducibility"]:
            raise ValueError("Seeds not set, please use Reproducer.set_seed() to do so or set ignore_reproducibility "
                             "to True in config")

        running_metric_results = {'loss': 0}
        for name in self.metrics.keys():
            running_metric_results[name] = 0

        epoch_start = self.__epoch
        for self.__epoch in range(epoch_start, _config["epochs"]):
            if _config["unfreeze_after"] and self.__epoch == _config["unfreeze_after"]:
                self.__unfreeze_model__()

            with tqdm(self.__train_dataloader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {self.__epoch}")
                for i, (inputs, y) in enumerate(tepoch, 0):
                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward + backward + optimize
                    y_hat = self.model(inputs)
                    loss = self.loss_fn(y_hat, y)
                    loss.backward()
                    self.optimizer.step()

                    running_metric_results['loss'] += loss.item()
                    metric_results = {}
                    for name, metric in self.metrics.items():
                        metric_results[name] = metric(y_hat, y.int()).item()
                        running_metric_results[name] += metric_results[name]

                    tepoch.set_postfix(metric_results, loss=loss.item() / self.batch_size)

                    if _config["log_every_n_batches"]:
                        self.log_training(_config, _run, i, running_metric_results)

                    if _config["val_every_n_batches"]:
                        batches = (i + 1) + self.dl_len * self.__epoch
                        if batches % _config["val_every_n_batches"] == 0:
                            self.__validate__(step=batches * self.batch_size)

            if not _config["log_every_n_batches"]:
                self.log_training(_config, _run, i, running_metric_results)

            if not _config["val_every_n_batches"] and self.__validation_dataloader:
                batches_total = self.dl_len * (self.__epoch + 1)
                self.__validate__(step=batches_total * self.batch_size)

            self.__save__(name=f'checkpoint_{self.__epoch + 1}.pth')

        _log.info('Finished Training')
        self.clean_checkpoints()
        return self.results


    def log_training(self, _config, _run, i, running_metric_results):
        batches_per_epoch = self.dl_len
        batches_total = (i + 1) + batches_per_epoch * (self.__epoch)

        divider = None
        if not _config["log_every_n_batches"] and i + 1 == self.dl_len:
            divider = self.dl_len
        if _config["log_every_n_batches"] and batches_total % _config["log_every_n_batches"] == 0:
            divider = _config["log_every_n_batches"]

        if divider:
            _run.log_scalar("LR", self.optimizer.param_groups[0]['lr'])
            _run.log_scalar("loss", running_metric_results["loss"] / divider, batches_total * self.batch_size)
            running_metric_results["loss"] = 0
            for name, metric in self.metrics.items():
                _run.log_scalar(name, running_metric_results[name] / divider,
                                batches_total * self.batch_size)
                running_metric_results[name] = 0


    @trainer_ingredient.capture
    @torch.no_grad()
    def __validate__(self, _run, _log, step=None, prefix="val_"):
        if not step:
            raise ValueError("step is required for validation")

        self.model.eval()

        for metric in self.__val_metrics.values():
            metric.reset()

        loss = 0
        for x, y in self.__validation_dataloader:
            y_hat = self.model(x)
            loss = self.loss_fn(y_hat, y).item()
            for metric in self.__val_metrics.values():
                metric.update(y_hat, y.int())

        self.results["loss"] = loss / len(self.__validation_dataloader)
        _run.log_scalar(prefix + "loss", self.results["loss"], step)
        for name, metric in self.__val_metrics.items():
            self.results[name] = metric.compute().item() / len(self.__validation_dataloader)
            _run.log_scalar(prefix + name, self.results[name], step)
            metric.reset()

        # save in trained state!
        self.model.train()

        if self.__best_validation_loss is None or loss < self.__best_validation_loss:
            _log.info(f'found new best validation loss, {self.__best_validation_loss} vs. {loss}')
            self.__best_validation_loss = loss
            self.__save__(name='best.pth', loss=loss)


    @trainer_ingredient.capture
    def __get_final_cp_dir__(self, _run, cp_dir, cp_dir_append_experiment, cp_dir_append_run):
        if not cp_dir: return None

        if cp_dir_append_experiment:
            cp_dir = osp.join(cp_dir, _run.experiment_info['name'])
        if cp_dir_append_run:
            if not _run._id:
                return None
            cp_dir = osp.join(cp_dir, str(_run._id) if str(_run._id) != '' else '-')
        os.makedirs(cp_dir, exist_ok=True)

        return cp_dir


    @trainer_ingredient.capture
    def __save__(self, _log, _run, loss: float = None, name: str = None):
        if not name:
            raise ValueError('Name is required')

        cp_dir = self.__get_final_cp_dir__()
        if not cp_dir: return

        cp_dir = osp.join(cp_dir, name)

        _log.info(f'Saving checkpoint: {cp_dir}')

        torch.save({'epoch': self.__epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_loss': self.__best_validation_loss,
                    'current_loss': loss,
                    },
                   cp_dir, pickle_module=dill)


    def load(self, cp_dir: str) -> None:
        checkpoint = torch.load(cp_dir)
        self.__epoch = checkpoint['epoch'] + 1
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.__best_validation_loss = checkpoint['best_loss']


    @trainer_ingredient.capture
    def clean_checkpoints(self, remove_cp_after_training):
        cp_dir = self.__get_final_cp_dir__()
        if remove_cp_after_training and cp_dir:
            for f in os.listdir(cp_dir):
                if f != f'checkpoint_{self.__epoch + 1}.pth' and f != 'best.pth':
                    os.remove(osp.join(cp_dir, f))


    def __unfreeze_model__(self):
        for param in self.model.parameters():
            param.requires_grad = True
