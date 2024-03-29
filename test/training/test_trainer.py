import os
import shutil
import unittest
from unittest import mock
from unittest.mock import MagicMock

import torch
import wandb
from torch import Tensor
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassAccuracy

from pthelpers.training.persist import save_training_state
from test.mnist_test import MnistTest
from pthelpers.training import Trainer


class Dummy():
    pass


class Test(MnistTest):
    def __init__(self, methodName):
        super().__init__(methodName)

    def test_train(self):
        wandb.run.name = "test_train"
        wandb.config["training"].update({"val_interval_batches": 1, "cleanup_after_training": False})
        trainer = Trainer(self.train_loader, self.test_loader, 10, self.test_loader)

        trainer._Trainer__train_epoch = MagicMock(return_value=True)
        train_epoch = trainer._Trainer__train_epoch
        trainer.test = MagicMock(return_value=None)

        epochs = 5
        trainer.train(self.model, self.optimizer, epochs)
        assert train_epoch.call_count == epochs, f"Expected {epochs} calls for train_epoch, got {train_epoch.call_count}"
        # since we mocked the train epoch method
        assert trainer.test.call_count == 0, f"Expected zero calls for test, got {trainer.test.call_count}"


    def test_train_epoch(self):
        wandb.run.name = "test_train_epoch"
        trainer = Trainer(self.train_loader, self.test_loader, 10, self.test_loader)

        self.model.eval()
        assert not self.model.training

        self.optimizer.zero_grad = MagicMock()
        self.optimizer.step = MagicMock()

        loss = Dummy()
        trainer.loss_fn = MagicMock(return_value=loss)
        loss.backward = MagicMock()
        loss.item = MagicMock(return_value=1.23)

        trainer.metrics["acc"] = MulticlassAccuracy(10)
        trainer.metrics["acc"].update = MagicMock()
        trainer._Trainer__val_metrics["acc"] = MulticlassAccuracy(10)
        trainer._Trainer__val_metrics["acc"].update = MagicMock()

        trainer._Trainer__inter_epoch_training_log = MagicMock(return_value=True)
        trainer._Trainer__inter_epoch_validation = MagicMock()

        self.__prepare_trainer_for_direct_train(trainer)
        assert trainer._Trainer__train_epoch(self.model, self.optimizer)

        assert self.model.training
        assert self.optimizer.step.call_count == 2
        assert self.optimizer.zero_grad.call_count == 2
        assert trainer.loss_fn.call_count == 4
        assert loss.backward.call_count == 2
        assert trainer.metrics["acc"].update.call_count == 2
        assert trainer._Trainer__val_metrics["acc"].update.call_count == 2
        assert trainer._Trainer__inter_epoch_training_log.call_count == 2
        assert trainer._Trainer__inter_epoch_validation.call_count == 2
        assert trainer._Trainer__logging_infos["running_loss"] == 2.46


    def test_dry_run(self):
        wandb.run.name = "test_dry_run"
        wandb.config["training"].update({"dry_run": True})
        trainer = Trainer(self.train_loader, self.test_loader, 10, self.test_loader)

        self.optimizer.zero_grad = MagicMock()

        trainer._Trainer__inter_epoch_training_log = MagicMock(return_value=True)
        self.__prepare_trainer_for_direct_train(trainer)
        trainer._Trainer__train_epoch(self.model, self.optimizer)

        assert self.optimizer.zero_grad.call_count == 1


    def test_train_logging_interval(self):
        wandb.run.name = "test_train_logging_interval"
        trainer = Trainer(self.train_loader, self.test_loader, 10, self.test_loader)

        # 3 epochs with 2 batches each --> 2 logs
        wandb.config["training"].update({"log_interval_batches": 3})
        with mock.patch.object(trainer, '_Trainer__training_log') as log_fn:
            trainer.train(self.model, self.optimizer, 3)
            assert log_fn.call_count == 2


    def test_train_logging_end_of_epoch(self):
        wandb.run.name = "test_train_logging_end_of_epoch"
        trainer = Trainer(self.train_loader, self.test_loader, 10, self.test_loader)

        # one log per epoch
        wandb.config["training"].update({"log_interval_batches": None})
        with mock.patch.object(trainer, '_Trainer__training_log') as log_fn:
            trainer.train(self.model, self.optimizer, 3)
            assert log_fn.call_count == 3


    def test_train_logging(self):
        wandb.run.name = "test_train_logging"
        trainer = Trainer(self.train_loader, self.test_loader, 10, self.test_loader)

        # now test if real logging works
        wandb.config["training"].update({"val_interval_batches": 9999, "dry_run": True, "log_interval_batches": None})
        with mock.patch.object(wandb, 'log') as wandblog:
            trainer.metrics["acc"] = MulticlassAccuracy(10)
            computed = Dummy()
            trainer.metrics["acc"].compute = MagicMock(return_value=computed)
            computed.item = MagicMock(return_value=0.534)

            loss = Dummy()
            trainer.loss_fn = MagicMock(return_value=loss)
            loss.backward = MagicMock()
            loss.item = MagicMock(return_value=1.23)

            trainer.train(self.model, self.optimizer, 3)

            assert trainer._Trainer__logging_infos["running_loss"] == 0
            assert wandblog.call_count == 1, f"Expected three calls, got {wandb.log.call_count}"
            self.assertDictEqual(wandblog.call_args_list[0].args[0],
                                 {'training results/t_loss': 1.23, 'training results/lr': 0.001, 'training results/acc': 0.534,
                                  'Hidden Panels/epoch': 1, 'Hidden Panels/batch': 2, 'Hidden Panels/sample': 1000})


    def test_validation_interval(self):
        wandb.run.name = "test_validation_interval"
        trainer = Trainer(self.train_loader, self.test_loader, 10, self.test_loader)

        # 3 epochs with 2 batches each --> 2 vals
        wandb.config["training"].update({"val_interval_batches": 3})
        with mock.patch.object(trainer, '_Trainer__validate') as val_fn:
            trainer.train(self.model, self.optimizer, 3)
            assert val_fn.call_count == 2
        shutil.rmtree("checkpoints")


    def test_validation_end_of_epoch(self):
        wandb.run.name = "test_validation_end_of_epoch"
        trainer = Trainer(self.train_loader, self.test_loader, 10, self.test_loader)

        # one log per epoch
        wandb.config["training"].update({"val_interval_batches": None})
        with mock.patch.object(trainer, '_Trainer__validate') as val_fn:
            trainer.train(self.model, self.optimizer, 3)
            assert val_fn.call_count == 3


    def test_validation_logging(self):
        wandb.run.name = "test_validation_logging"
        trainer = Trainer(self.train_loader, self.test_loader, 10, self.test_loader)
        # now test if real logging works
        wandb.config["training"].update({"log_interval_batches": None, "dry_run": True, "val_interval_batches": None})
        with mock.patch.object(wandb, 'log') as wandblog:
            trainer.test = MagicMock(return_value=(0.123, Tensor([]), Tensor([])))
            trainer._Trainer__val_metrics["acc"] = MulticlassAccuracy(10)
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
            assert wandblog.call_count == 2, f"Expected four calls, got {wandb.log.call_count}"
            self.assertDictEqual(wandblog.call_args_list[1].args[0],
                                 {'validation results/v_loss': 0.123, 'validation results/v_acc': 0.534, 'Hidden Panels/epoch': 1, 'Hidden Panels/batch': 2, 'Hidden Panels/sample': 1000})


    def test_cleanup(self):
        wandb.run.name = "test_cleanup"
        wandb.config["training"].update({"cleanup_after_training": True})
        trainer = Trainer(self.train_loader, self.test_loader, 10, self.test_loader)

        epochs = 3
        trainer.train(self.model, self.optimizer, epochs)
        cp_dir = os.path.join("checkpoints", "test_cleanup")
        # 3rd epoch and best val
        files = os.listdir(cp_dir)
        assert len(files) == 2, f"Expected {epochs} epoch checkpoints, got {len(files)}"
        assert "best.pth" in files
        assert "final.pth" in files

    def test_warm_start(self):
        wandb.run.name = "test_warm_start"
        cp_dir = os.path.join("checkpoints", "test_warm_start")
        wandb.config["training"].update({"cleanup_after_training": False, "val_interval_batches": 999})
        trainer = Trainer(self.train_loader, self.test_loader, 10, self.test_loader)

        epochs = 1
        trainer._Trainer__train_epoch = MagicMock()
        trainer.train(self.model, self.optimizer, epochs)
        save_training_state(trainer, self.model, self.optimizer)

        # one epoch
        files = os.listdir(cp_dir)
        assert trainer._Trainer__train_epoch.call_count == epochs
        assert len(files) == 1, f"Expected {epochs} epoch checkpoints, got {len(files)}"

        # add one epoch
        epochs = 2
        trainer.train(self.model, self.optimizer, epochs)
        save_training_state(trainer, self.model, self.optimizer)
        assert trainer._Trainer__train_epoch.call_count == epochs
        files = os.listdir(cp_dir)
        assert len(files) == 2, f"Expected {epochs} epoch checkpoints, got {len(files)}"

        # add nothing
        trainer.train(self.model, self.optimizer, epochs)
        assert trainer._Trainer__train_epoch.call_count == epochs
        files = os.listdir(cp_dir)
        assert len(files) == 2, f"Expected {epochs} epoch checkpoints, got {len(files)}"


    def __prepare_trainer_for_direct_train(self, trainer: Trainer):
        trainer._Trainer__reset()
        trainer._Trainer__logging_infos["end_epoch"] = 999
        trainer._Trainer__logging_infos["running_loss"] = 0
        trainer.device = torch.device("cpu")

    def test_unfreeze(self):
        wandb.run.name = "test_unfreeze"
        wandb.config["training"].update({"cleanup_after_training": False, "unfreeze_after": 2})
        trainer = Trainer(self.train_loader, self.test_loader, 10, self.test_loader)

        self.model.parameters = MagicMock(return_value=[])
        trainer.train(self.model, self.optimizer, 1)
        assert self.model.parameters.call_count == 0, "expected 0 unfreeze attempts"

        trainer.train(self.model, self.optimizer, 2)
        assert self.model.parameters.call_count == 1, "expected 1 unfreeze attempts"



if __name__ == '__main__':
    unittest.main()
