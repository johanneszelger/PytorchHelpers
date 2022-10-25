import os
import shutil
import unittest
import time
from unittest import mock
from unittest.mock import MagicMock

import torch
import wandb
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision import transforms, datasets

from models.simple_net import SimpleNet
from src.training.trainer import Trainer


class Dummy():
    pass


class Test(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_data = datasets.MNIST('../data', train=True, download=True,
                                    transform=transform)
        train_data.data = train_data.data[:1000]
        test_data = datasets.MNIST('../data', train=False,
                                   transform=transform)
        test_data.data = test_data.data[:1000]

        self.train_loader = DataLoader(train_data, batch_size=500)
        self.test_loader = DataLoader(test_data, batch_size=500)
        self.model = SimpleNet()
        self.optimizer = Adam(self.model.parameters())

    def setUp(self) -> None:
        wandb.init(mode="disabled")
        wandb.config.update({"cp_base_path": "checkpoints"})


    def tearDown(self) -> None:
        time.sleep(0.5)
        shutil.rmtree(wandb.config["cp_base_path"], ignore_errors=True)


    def test_train(self):
        wandb.run.name = "test_train"
        wandb.config.update({"val_interval_batches": 1})
        trainer = Trainer(self.train_loader, self.test_loader, self.test_loader)

        trainer._Trainer__train_epoch = MagicMock(return_value=True)
        train_epoch = trainer._Trainer__train_epoch
        trainer.test = MagicMock(return_value=None)

        epochs = 5
        trainer.train(self.model, self.optimizer, epochs)
        assert train_epoch.call_count == epochs, f"Expected {epochs} calls for train_epoch, got {train_epoch.call_count}"
        # since we mocked the train epoch method
        assert trainer.test.call_count == 0, f"Expected zero calls for test, got {trainer.test.call_count}"

        cp_dir = os.path.join("checkpoints", "test_train")
        files = os.listdir(cp_dir)
        assert len(files) == epochs, f"Expected {epochs} epoch checkpoints, got {len(files)}"


    def test_train_epoch(self):
        wandb.run.name = "test_train_epoch"
        trainer = Trainer(self.train_loader, self.test_loader, self.test_loader)

        self.model.eval()
        assert not self.model.training

        self.optimizer.zero_grad = MagicMock()
        self.optimizer.step = MagicMock()

        loss = Dummy()
        trainer.loss_fn = MagicMock(return_value=loss)
        loss.backward = MagicMock()
        loss.item = MagicMock(return_value=1.23)

        trainer.metrics["acc"] = Accuracy()
        trainer.metrics["acc"].update = MagicMock()

        trainer._Trainer__inter_epoch_training_log = MagicMock(return_value=True)
        trainer._Trainer__inter_epoch_validation = MagicMock()

        self.__prepare_trainer_for_direct_train(trainer)
        trainer._Trainer__train_epoch(self.model, self.optimizer)

        assert self.model.training
        assert self.optimizer.step.call_count == 2
        assert self.optimizer.zero_grad.call_count == 2
        assert trainer.loss_fn.call_count == 2
        assert loss.backward.call_count == 2
        assert trainer.metrics["acc"].update.call_count == 2
        assert trainer._Trainer__inter_epoch_training_log.call_count == 2
        assert trainer._Trainer__inter_epoch_validation.call_count == 2
        assert trainer._Trainer__logging_infos["running_loss"] == 2.46


    def test_dry_run(self):
        wandb.run.name = "test_dry_run"
        wandb.config.update({"dry_run": True})
        trainer = Trainer(self.train_loader, self.test_loader, self.test_loader)

        self.optimizer.zero_grad = MagicMock()

        trainer._Trainer__inter_epoch_training_log = MagicMock(return_value=True)
        self.__prepare_trainer_for_direct_train(trainer)
        trainer._Trainer__train_epoch(self.model, self.optimizer)

        assert self.optimizer.zero_grad.call_count == 1


    def test_train_logging(self):
        wandb.run.name = "test_train_logging"
        trainer = Trainer(self.train_loader, self.test_loader, self.test_loader)

        # 3 epochs with 2 batches each --> 2 logs
        wandb.config.update({"log_interval_batches": 3}, allow_val_change=True)
        with mock.patch.object(trainer, '_Trainer__training_log') as log_fn:
            trainer.train(self.model, self.optimizer, 3)
            assert log_fn.call_count == 2

        # one log per epoch
        wandb.config.update({"log_interval_batches": None}, allow_val_change=True)
        with mock.patch.object(trainer, '_Trainer__training_log') as log_fn:
            trainer.train(self.model, self.optimizer, 3)
            assert log_fn.call_count == 3

        # now test if real logging works
        wandb.config.update({"val_interval_batches": 9999, "dry_run": True}, allow_val_change=True)
        with mock.patch.object(wandb, 'log') as wandblog:
            trainer.metrics["acc"] = Accuracy()
            computed = Dummy()
            trainer.metrics["acc"].compute = MagicMock(return_value=computed)
            computed.item = MagicMock(return_value=0.534)

            loss = Dummy()
            trainer.loss_fn = MagicMock(return_value=loss)
            loss.backward = MagicMock()
            loss.item = MagicMock(return_value=1.23)

            trainer.train(self.model, self.optimizer, 3)

            assert trainer._Trainer__logging_infos["running_loss"] == 0
            assert wandblog.call_count == 1, f"Expected one call, got {wandb.log.call_count}"
            self.assertDictEqual(wandblog.call_args_list[0].args[0],
                                 {'t_loss': 1.23, 'lr': 0.001, 'acc': 0.534, 'epoch': 1, 'batch': 2, 'sample': 1000})


    def test_validation(self):
        wandb.run.name = "test_validation"
        trainer = Trainer(self.train_loader, self.test_loader, self.test_loader)

        # 3 epochs with 2 batches each --> 2 vals
        wandb.config.update({"val_interval_batches": 3}, allow_val_change=True)
        with mock.patch.object(trainer, '_Trainer__validate') as val_fn:
            trainer.train(self.model, self.optimizer, 3)
            assert val_fn.call_count == 2

        # one log per epoch
        wandb.config.update({"val_interval_batches": None}, allow_val_change=True)
        with mock.patch.object(trainer, '_Trainer__validate') as val_fn:
            trainer.train(self.model, self.optimizer, 3)
            assert val_fn.call_count == 3

        # now test if real logging works
        wandb.config.update({"log_interval_batches": None, "dry_run": True}, allow_val_change=True)
        with mock.patch.object(wandb, 'log') as wandblog:
            trainer.test = MagicMock(return_value=0.123)
            trainer._Trainer__val_metrics["acc"] = Accuracy()
            computed = Dummy()
            trainer._Trainer__val_metrics["acc"].compute = MagicMock(return_value=computed)
            computed.item = MagicMock(return_value=0.534)

            loss = Dummy()
            trainer.loss_fn = MagicMock(return_value=loss)
            loss.backward = MagicMock()
            loss.item = MagicMock(return_value=1.23)

            trainer.train(self.model, self.optimizer, 3)

            assert trainer.test.call_count == 1, f"Expected one call, got {trainer.test.call_count}"
            # once for training log once for val
            assert wandblog.call_count == 2, f"Expected two calls, got {wandb.log.call_count}"
            self.assertDictEqual(wandblog.call_args_list[1].args[0],
                                 {'v_loss': 0.123, 'acc': 0.534, 'epoch': 1, 'batch': 2, 'sample': 1000})


    def test_cleanup(self):
        wandb.run.name = "test_cleanup"


    def test_warm_start(self):
        wandb.run.name = "test_warm_start"


    def __prepare_trainer_for_direct_train(self, trainer: Trainer):
        trainer._Trainer__reset()
        trainer._Trainer__logging_infos["end_epoch"] = 999
        trainer._Trainer__logging_infos["running_loss"] = 0
        trainer.device = torch.device("cpu")


if __name__ == '__main__':
    unittest.main()
