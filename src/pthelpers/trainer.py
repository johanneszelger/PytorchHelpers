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
    epochs = 1
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
        if not osp.exists(cp_dir):
            return None

        epochs = [x.replace("checkpoint_", "").replace(".pth", "") for x in os.listdir(cp_dir) if not x.startswith("best")]
        if len(epochs) == 0:
            return None
        last_epoch = max(epochs)

        _log.info(f'Found existing checkpoint: checkpoint_{last_epoch}.pth in {cp_dir}')
        self.load(osp.join(cp_dir, f"checkpoint_{last_epoch}.pth"))


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
                 ) -> None:
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

        self.__model = model
        self.__train_dataloader = train_dataloader
        self.__validation_dataloader = validation_dataloader
        self.__loss_fn = loss_fn
        self.__optimizer = optimizer
        self.__metrics = metrics
        self.__val_metrics = copy.deepcopy(metrics)

        self.__best_validation_loss = None
        self.__epoch = 0


    @trainer_ingredient.capture()
    def train(self, _log, _run: Run, _config) -> None:
        """
        starts the training of the model
        :param _run: sacred experiment run
        :param _config: sacred experiment config
        :return: None
        """

        self.__load_existing_cp__()

        _log.info(f'Starting training')
        if not Reproducer.seed_set and not _config["ignore_reproducibility"]:
            raise ValueError("Seeds not set, please use Reproducer.set_seed() to do so or set ignore_reproducibility "
                             "to True in config")

        # load existing here

        # send all to device here
        device = "cuda" if torch.cuda.is_available() and _config["use_gpu"] else "cpu"
        self.__model.to(device)
        self.__loss_fn.to(device)
        for metric in self.__metrics.values():
            metric.to(device)
        for metric in self.__val_metrics.values():
            metric.to(device)

        samples_per_log = _config["log_every_n_batches"] * self.__train_dataloader.batch_size if _config["log_every_n_batches"] else 1
        batch_size = self.__train_dataloader.batch_size
        epoch_start = self.__epoch
        for self.__epoch in range(epoch_start, _config["epochs"]):
            running_loss = 0.0
            running_metric_results = {}
            for name in self.__metrics.keys():
                running_metric_results[name] = 0

            with tqdm(self.__train_dataloader, unit="batch") as tepoch:
                for i, (inputs, y) in enumerate(tepoch, 0):
                    tepoch.set_description(f"Epoch {self.__epoch}")

                    # zero the parameter gradients
                    self.__optimizer.zero_grad()

                    # forward + backward + optimize
                    y_hat = self.__model(inputs)
                    loss = self.__loss_fn(y_hat, y)
                    loss.backward()
                    self.__optimizer.step()

                    running_loss += loss.item()

                    metric_results = {}
                    for name, metric in self.__metrics.items():
                        metric_results[name] = metric(y_hat, y.int()).item()
                        running_metric_results[name] += metric_results[name]

                    tepoch.set_postfix(metric_results, loss=loss.item() / batch_size)

                    if _config["log_every_n_batches"]:
                        batches = (i + 1) + len(self.__train_dataloader) * self.__epoch
                        if batches % _config["log_every_n_batches"] == 0:
                            _run.log_scalar("loss", running_loss / samples_per_log, batches * batch_size)
                            running_loss = 0.0
                            for name, metric in self.__metrics.items():
                                _run.log_scalar(name, running_metric_results[name] / _config["log_every_n_batches"],
                                                batches * batch_size)
                                running_metric_results[name] = 0

                    if _config["val_every_n_batches"]:
                        batches = (i + 1) + len(self.__train_dataloader) * self.__epoch
                        if batches % _config["val_every_n_batches"] == 0:
                            self.__validate__(step=batches * batch_size)

            if _config["log_every_n_batches"] is None:
                batches = len(self.__train_dataloader) * (self.__epoch + 1)
                samples = len(self.__train_dataloader) * batch_size
                _run.log_scalar("loss", running_loss / samples, batches * batch_size)
                for name, metric in self.__metrics.items():
                    _run.log_scalar(name, running_metric_results[name] / _config["log_every_n_batches"],
                                    batches * self.__train_dataloader.batch_size)

            if _config["val_every_n_batches"] is None and self.__validation_dataloader:
                batches = len(self.__train_dataloader) * (self.__epoch + 1)
                self.__validate__(step=batches * self.__train_dataloader.batch_size)

            self.__save__(name=f'checkpoint_{self.__epoch}.pth')

        _log.info('Finished Training')
        return


    @trainer_ingredient.capture
    @torch.no_grad()
    def __validate__(self, _run, _log, step=None, prefix="val_"):
        if not step:
            raise ValueError("step is required for validation")

        for metric in self.__val_metrics.values():
            metric.reset()

        loss = 0
        for x, y in self.__validation_dataloader:
            y_hat = self.__model(x)
            loss = self.__loss_fn(y_hat, y).item()
            for metric in self.__val_metrics.values():
                metric.update(y_hat, y.int())

        _run.log_scalar(prefix + "loss", loss / len(self.__validation_dataloader), step)
        for name, metric in self.__val_metrics.items():
            _run.log_scalar(prefix + name, metric.compute().item() / len(self.__validation_dataloader), step)
            metric.reset()

        if self.__best_validation_loss is None or loss < self.__best_validation_loss:
            _log.info(f'found new best validation loss, {self.__best_validation_loss} vs. {loss}')
            self.__best_validation_loss = loss
            self.__save__(name='best.pth', loss=loss)


    #
    #
    # @staticmethod
    # def test(data_loader: DataLoader, model: Module, checkpoint: str = None, use_gpu: bool = True):
    #
    #     if checkpoint:
    #         Trainer.load_model_weights(model, checkpoint)
    #
    #     device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
    #     model.to(device)
    #
    #     gt = torch.IntTensor().to(device)
    #     pred = torch.FloatTensor().to(device)
    #
    #     model.eval()
    #     with torch.no_grad():
    #         for i, (X, target) in enumerate(data_loader):
    #             target = target.to(device)
    #             gt = torch.cat((gt, target), 0)
    #
    #             bs, c, h, w = X.size()
    #             X = X.view(-1, c, h, w).to(device)
    #
    #             out = model(X)
    #             pred = torch.cat((pred, out), 0)
    #
    #     return gt, pred


    @trainer_ingredient.capture
    def __get_final_cp_dir__(self, _run, cp_dir, cp_dir_append_experiment, cp_dir_append_run):
        if cp_dir_append_experiment:
            cp_dir = osp.join(cp_dir, _run.experiment_info['name'])
        if cp_dir_append_run:
            cp_dir = osp.join(cp_dir, str(_run._id))
        os.makedirs(cp_dir, exist_ok=True)

        return cp_dir


    @trainer_ingredient.capture
    def __save__(self, _log, _run, loss: float = None, name: str = None):
        if not name:
            raise ValueError('Name is required')

        cp_dir = self.__get_final_cp_dir__()
        cp_dir = osp.join(cp_dir, name)

        _log.info(f'Saving checkpoint: {cp_dir}')

        torch.save({'epoch': self.__epoch,
                    'state_dict': self.__model.state_dict(),
                    'optimizer': self.__optimizer.state_dict(),
                    'best_loss': self.__best_validation_loss,
                    'current_loss': loss,
                    },
                   cp_dir, pickle_module=dill)


    def load(self, cp_dir: str) -> None:
        checkpoint = torch.load(cp_dir)
        self.__epoch = checkpoint['epoch']
        self.__model.load_state_dict(checkpoint['state_dict'])
        self.__optimizer.load_state_dict(checkpoint['optimizer'])
        self.__best_validation_loss = checkpoint['best_loss']


    # @staticmethod
    # def load_model_weights(model: Module, cp_path: str):
    #     checkpoint = torch.load(cp_path)
    #     model.load_state_dict(checkpoint['state_dict'])
    #
    #
    # def __init_writers__(self):
    #     self.writer_train = SummaryWriter(osp.join(self.log_dir, TRAIN_LOG_SUBDIR)) if self.log_dir else None
    #     self.writer_val = SummaryWriter(
    #             osp.join(self.log_dir, VAL_LOG_SUBDIR)) if self.log_dir and self.__validation_dataloader else None
    #
    #
    # def __close_writers__(self):
    #     if self.writer_train:
    #         self.writer_train.close()
    #     if self.writer_val:
    #         self.writer_val.close()
    #
    #
    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     # Remove the unpickable entries.
    #     del state['writer_train']
    #     del state['writer_val']
    #     return state
