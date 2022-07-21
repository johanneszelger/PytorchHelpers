""" Providing Trainer that automates pthelpers training """

import torch
from sacred import Ingredient
from sacred.run import Run
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from pthelpers.reproducibility import Reproducer
except:
    from src.pthelpers.reproducibility import Reproducer

trainer_ingredient = Ingredient('trainer')


@trainer_ingredient.config
def cfg():
    cp_dir = None
    epochs = 1
    use_gpu = True
    log_every_n_batches = None
    validate_every_n_samples = None
    ignore_reproducibility = False


class Trainer:
    """
    Trainer that automates the training of models including features like automated checkpoint saving
    and logging
    """


    # @trainer_ingredient.capture()
    # def load_latest_existing(self, cp_dir: str):
    #     if not osp.exists(cp_dir):
    #         return None
    #     epochs = [x.replace("checkpoint_", "").replace(".pth", "") for x in os.listdir(cp_dir) if not x.startswith("best")]
    #     if len(epochs) == 0:
    #         return None
    #     last_epoch = max(epochs)
    #     return self.load(osp.join(cp_dir, f"checkpoint_{last_epoch}.pth"))

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
            metrics = {}

        self.__model = model
        self.__train_dataloader = train_dataloader
        self.__loss_fn = loss_fn
        self.__optimizer = optimizer
        self.__metrics = metrics


        # self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        # for metric in self.metrics.values():
        #     metric.to(self.device)
        # self.checkpoints_dir = checkpoints_dir
        # self.log_dir = log_dir
        # self.__validation_dataloader = validation_dataloader
        # self.validate_every_steps = validate_every_steps
        # self.remove_cp_after_training = remove_cp_after_training
        # self.eval_first = eval_first
        # self.class_names = class_names
        # self.__use_gpu = use_gpu

        # self.writer_train = None
        # self.writer_val = None

        # self.__best_validation_loss__ = None
        # self.__current_loss__ = None
        #
        # if self.checkpoints_dir and not osp.exists(self.checkpoints_dir):
        #     os.makedirs(self.checkpoints_dir)
        #
        # if self.log_dir and osp.exists(self.log_dir) and len(os.listdir(self.log_dir)) > 0:
        #     delete = input(f"Log directory not empty ({self.log_dir}), should it be emptied? (N)?")
        #     if delete == 'Y' or delete == 'y' or delete == 'yes' or delete == 'Yes':
        #         shutil.rmtree(self.log_dir)
        #
        # if self.log_dir and not osp.exists(self.log_dir):
        #     os.makedirs(self.log_dir)
        #
        # self.current_epoch = 0
        #
        # if load_latest_existing:
        #     if not checkpoints_dir:
        #         raise ValueError("If latest existing checkpoint should be loaded, please specify checkpoints_dir")
        #     self.load_latest_existing(checkpoints_dir)


    @trainer_ingredient.capture()
    def train(self, _run: Run, _config) -> None:
        """
        starts the training of the model
        :param _run: sacred experiment run
        :param _config: sacred experiment config
        :return: None
        """
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

        samples_per_batch = _config["log_every_n_batches"] * self.__train_dataloader.batch_size
        epoch_start = 0
        for epoch in range(epoch_start, _config["epochs"]):

            running_loss = 0.0
            with tqdm(self.__train_dataloader, unit="batch") as tepoch:
                for i, (inputs, labels) in enumerate(tepoch, 0):
                    tepoch.set_description(f"Epoch {epoch}")

                    # zero the parameter gradients
                    self.__optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = self.__model(inputs)
                    loss = self.__loss_fn(outputs, labels)
                    loss.backward()
                    self.__optimizer.step()

                    running_loss += loss.item()

                    if _config["log_every_n_batches"]:
                        batches = (i + 1) + len(self.__train_dataloader) * epoch
                        if batches % _config["log_every_n_batches"]:  # every time the log has been surpassed
                            tepoch.set_postfix(loss=running_loss, )#accuracy=100. * accuracy)
                            _run.log_scalar("loss", running_loss / samples_per_batch, batches)
                            running_loss = 0.0

        print('Finished Training')
        return

        # eval model with random params
        # if start_at_epoch == 0 and self.eval_first:
        #     print("doing evaluation before training")
        #     self.__validate__(0, self.__train_dataloader)
        #     self.__validate__(0, self.__validation_dataloader)
        #
        # step = start_at_epoch * len(self.__train_dataloader)
        #
        # # keep track of results
        # targets_full = []
        # outputs_full = []
        #
        # for epoch in range(start_at_epoch, until_epoch):
        #     self.current_epoch += 1
        #     # Print epoch
        #     print(f'Starting epoch {self.current_epoch}')
        #
        #     # Iterate over the DataLoader for training data
        #     for _, data in enumerate(self.__train_dataloader, 0):
        #         step += 1
        #
        #         # Get inputs
        #         inputs, targets = data
        #
        #         # Zero the gradients
        #         self.optimizer.zero_grad()
        #
        #         # Perform forward pass
        #         outputs = self.model(inputs)
        #
        #         targets_full += [targets]
        #         outputs_full += [outputs]
        #
        #         # Compute loss
        #         loss = self.loss_fn(outputs, targets)
        #
        #         # Perform backward pass
        #         loss.backward()
        #
        #         # Perform optimization
        #         self.optimizer.step()
        #
        #         # print/log train and val
        #         if self.validate_every_steps and step % self.validate_every_steps == 0:
        #             outputs_full = torch.cat(outputs_full, dim=0)
        #             targets_full = torch.cat(targets_full, dim=0)
        #             self.__calc_metrics_print_save__(outputs_full, targets_full, step, False)
        #             outputs_full = []
        #             targets_full = []
        #             self.__validate__(step)
        #
        #     if self.checkpoints_dir:
        #         self.__save__(osp.join(self.checkpoints_dir, f'checkpoint_{self.current_epoch}.pth'))
        #
        # # final eval at end of training
        # if step % self.validate_every_steps != 0:
        #     self.__validate__(step, self.__train_dataloader)
        #     self.__validate__(step, self.__validation_dataloader)
        #
        # if self.remove_cp_after_training and self.checkpoints_dir:
        #     for f in os.listdir(self.checkpoints_dir):
        #         if f != f'checkpoint_{until_epoch}.pth' and f != 'best.pth':
        #             os.remove(osp.join(self.checkpoints_dir, f))
        #
        # self.__close_writers__()


    # def __validate__(self, step: int, dataloader: DataLoader = None) -> None:
    #     if dataloader is None:
    #         dataloader = self.__validation_dataloader
    #
    #     if dataloader is None:
    #         return
    #
    #     gt, pred = Trainer.test(dataloader, model=self.model, use_gpu=self.__use_gpu)
    #
    #     self.model.train()
    #
    #     self.__calc_metrics_print_save__(pred, gt, step,
    #                                      dataloader == self.__validation_dataloader)
    #
    #
    # def __calc_metrics_print_save__(self, predictions: torch.Tensor, targets: torch.Tensor, step: int,
    #                                 validation: bool) -> None:
    #     class_names = self.class_names if self.class_names is not None \
    #         else [str(x) for x in np.arange(1, predictions.shape[1] + 1, 1)]
    #
    #     with torch.no_grad():
    #         loss = self.loss_fn(predictions, targets).item()  # / divider
    #         if not validation:
    #             self.__current_loss__ = loss
    #
    #         writer = self.writer_val if validation else self.writer_train
    #
    #         if writer:
    #             writer.add_scalar('Loss', loss, step, time.time())
    #             if not validation:
    #                 writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], step, time.time())
    #         print(f'{"val_" if validation else "train_"}loss %5d: %.3f' % (step, loss))
    #
    #         if self.metrics:
    #             for metric_name, metric in self.metrics.items():
    #                 value = metric(predictions, targets.int())  # / divider
    #                 if isinstance(value, int) or value.dim() == 0:
    #                     if writer:
    #                         writer.add_scalar(f'{metric_name}', value, step, time.time())
    #                     print(f'{"val_" if validation else "train_"}{metric_name} %5d: %.3f' % (step, value))
    #                 else:
    #                     for x in zip(class_names, value):
    #                         if writer:
    #                             writer.add_scalar(f'{metric_name} {x[0]}', x[1], step, time.time())
    #                         print(f'{"val_" if validation else "train_"}{metric_name} {x[0]} %5d: %s' % (step, x[1]))
    #
    #         # add PR curves
    #         if validation and writer:
    #             gt = np.eye(predictions.cpu().shape[1], dtype='uint8')[targets.cpu()] if targets.dim() == 1 \
    #                 else targets
    #             for i, c in enumerate(class_names):
    #                 writer.add_pr_curve(f'{c} PR', gt[:, i], predictions[:, i], step)
    #
    #         if self.checkpoints_dir and validation and step > 0 and \
    #                 (self.__best_validation_loss__ is None or loss < self.__best_validation_loss__):
    #             self.__best_validation_loss__ = loss
    #             self.__save__(osp.join(self.checkpoints_dir, 'best.pth'))
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
    #
    #
    # def __save__(self, dir: str):
    #     torch.save({'epoch': self.current_epoch,
    #                 'state_dict': self.model.state_dict(),
    #                 'optimizer': self.optimizer.state_dict(),
    #                 'best_loss': self.__best_validation_loss__,
    #                 'current_loss': self.__current_loss__,
    #                 },
    #                dir, pickle_module=dill)
    #
    #
    # def load(self, dir: str):
    #     checkpoint = torch.load(dir)
    #     self.current_epoch = checkpoint['epoch']
    #     self.model.load_state_dict(checkpoint['state_dict'])
    #     self.optimizer.load_state_dict(checkpoint['optimizer'])
    #     self.__best_validation_loss__ = checkpoint['best_loss']
    #     self.__current_loss__ = checkpoint['current_loss']
    #
    #
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
