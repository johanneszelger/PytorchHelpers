import os
import os.path as osp
import shutil
import time
import unittest
from unittest.mock import MagicMock

import torch.optim
import torchvision.models
from sacred import Experiment
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from unittest import mock

from src.pthelpers.models.simple_net import Net
from src.pthelpers.reproducibility import Reproducer
from src.pthelpers.trainer import Trainer, trainer_ingredient


def optionHook(options):
    options['--debug'] = True
    options['--pdb'] = True


class TrainerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = Net()

        self.data, _ = torch.utils.data.random_split(
                torchvision.datasets.MNIST("./data/", download=True, transform=ToTensor()),
                [4096, 60000 - 4096])
        self.test_data = torchvision.datasets.MNIST("./data/", train=False, transform=ToTensor())

        self.dataloader = DataLoader(self.data, batch_size=1024, shuffle=True)
        self.test_dataloader = DataLoader(self.test_data, batch_size=1024, shuffle=False)

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), 1e-3)

        self.cp_dir = 'checkpoints'
        if osp.exists(self.cp_dir):
            shutil.rmtree(self.cp_dir)
        os.makedirs(self.cp_dir)

        self.experiment = Experiment(ingredients=[trainer_ingredient])
        self.experiment.option_hook(optionHook)
        trainer_ingredient.add_config({"use_gpu": False, "ignore_reproducibility": True})


    def tearDown(self) -> None:
        time.sleep(0.3)
        shutil.rmtree(self.cp_dir, ignore_errors=True)


    def test_init(self):
        trainer = Trainer(self.model, self.dataloader, self.test_dataloader, self.loss_fn, self.optimizer)
        self.assertIsNotNone(trainer)


    def test_train(self):
        @self.experiment.main
        def run():
            self.optimizer.step = MagicMock(return_value=None)
            trainer = Trainer(self.model, self.dataloader, self.test_dataloader, self.loss_fn, self.optimizer)
            trainer.train()
            self.optimizer.step.assert_called()


        self.experiment.run()


    def test_logging_vanilla(self):
        @self.experiment.main
        def run(_run):
            trainer = Trainer(self.model, self.dataloader, self.test_dataloader, self.loss_fn, self.optimizer)
            with mock.patch.object(_run, 'log_scalar') as writer:
                trainer.train()
                # Loss LR (train only) and Acc after epoch for train and val
                self.assertEqual(5, writer.call_count)


        self.experiment.run()


    def test_logging_every_n_batch(self):
        @self.experiment.main
        def run(_run):
            trainer = Trainer(self.model, self.dataloader, self.test_dataloader, self.loss_fn, self.optimizer)
            with mock.patch.object(_run, 'log_scalar') as writer:
                trainer.train()
                # twice Loss LR and Acc plus once val_loss and val_acc at end
                self.assertEqual(3 + 3 + 2, writer.call_count)


        self.experiment.run(config_updates={"trainer.log_every_n_batches": 2})


    def test_val_every_n_batch(self):
        @self.experiment.main
        def run(_run):
            trainer = Trainer(self.model, self.dataloader, self.test_dataloader, self.loss_fn, self.optimizer)
            with mock.patch.object(_run, 'log_scalar') as writer:
                trainer.train()
                # twice vall_loss and val_acc plus once Loss LR and Ac at end
                self.assertEqual(2 + 2 + 3, writer.call_count)


        self.experiment.run(config_updates={"trainer.val_every_n_batches": 2})


    def test_checkpoint_creation(self):
        @self.experiment.main
        def run(_run):
            trainer = Trainer(self.model, self.dataloader, self.test_dataloader, self.loss_fn, self.optimizer)
            trainer.train()

            self.assertTrue(osp.exists(osp.join(self.cp_dir, "checkpoint_1.pth")))
            self.assertTrue(osp.exists(osp.join(self.cp_dir, "checkpoint_2.pth")))
            self.assertTrue(osp.exists(osp.join(self.cp_dir, "checkpoint_3.pth")))
            self.assertTrue(osp.exists(osp.join(self.cp_dir, "best.pth")))


        self.experiment.run(config_updates={"trainer.epochs": 3, "trainer.cp_dir": self.cp_dir, "trainer.cp_dir_append_run": False,
                                            "trainer.remove_cp_after_training": False})


    def test_checkpoint_cleanup(self):
        @self.experiment.main
        def run(_run):
            trainer = Trainer(self.model, self.dataloader, self.test_dataloader, self.loss_fn, self.optimizer)
            trainer.train()

            self.assertFalse(osp.exists(osp.join(self.cp_dir, "checkpoint_1.pth")))
            self.assertFalse(osp.exists(osp.join(self.cp_dir, "checkpoint_2.pth")))
            self.assertTrue(osp.exists(osp.join(self.cp_dir, "checkpoint_3.pth")))
            self.assertTrue(osp.exists(osp.join(self.cp_dir, "best.pth")))


        self.experiment.run(config_updates={"trainer.epochs": 3, "trainer.cp_dir": self.cp_dir, "trainer.cp_dir_append_run": False})


    def test_save_and_continue(self):
        @self.experiment.main
        def run(_run):
            trainer = Trainer(self.model, self.dataloader, self.test_dataloader, __constant_loss__(), self.optimizer)
            trainer.train()


        self.experiment.run(config_updates={"trainer.cp_dir": self.cp_dir, "trainer.cp_dir_append_run": False})


        @self.experiment.main
        def run(_run):
            trainer = Trainer(self.model, self.dataloader, self.test_dataloader, __constant_loss__(2), self.optimizer)
            trainer.load(os.path.join(self.cp_dir, "checkpoint_1.pth"))
            with mock.patch('torch.save') as save:
                trainer.train()
                self.assertEqual(1, save.call_count)
                self.assertEqual(os.path.join(self.cp_dir, "checkpoint_2.pth"), save.call_args[0][1])


        self.experiment.run(config_updates={"trainer.epochs": 2, "trainer.cp_dir": self.cp_dir, "trainer.cp_dir_append_run": False,
                                            "trainer.remove_cp_after_training": False})


        @self.experiment.main
        def run(_run):
            trainer = Trainer(self.model, self.dataloader, self.test_dataloader, __constant_loss__(0), self.optimizer)
            trainer.loss_fn = __constant_loss__(0)
            with mock.patch('torch.save') as save:
                trainer.train()
                self.assertEqual(2, save.call_count)
                self.assertEqual(os.path.join(self.cp_dir, "best.pth"), save.call_args_list[0][0][1])
                self.assertEqual(os.path.join(self.cp_dir, "checkpoint_2.pth"), save.call_args_list[1][0][1])


        # only train to two epochs again, because the patch used earlier does not really save anything
        self.experiment.run(config_updates={"trainer.epochs": 2, "trainer.cp_dir": self.cp_dir, "trainer.cp_dir_append_run": False})


    def test_save_and_auto_load(self):
        @self.experiment.main
        def run(_run):
            trainer = Trainer(self.model, self.dataloader, self.test_dataloader, self.loss_fn, self.optimizer)
            trainer.train()

            self.optimizer.step = MagicMock(return_value=None)
            trainer = Trainer(self.model, self.dataloader, self.test_dataloader, self.loss_fn, self.optimizer)
            trainer.train()
            self.optimizer.step.assert_not_called()

        self.experiment.run(config_updates={"trainer.epochs": 2, "trainer.cp_dir": self.cp_dir, "trainer.cp_dir_append_run": False})
    #
    #
    # def test_test(self):
    #     gt, pred = Trainer.test(self.test_dataloader, self.model)
    #     self.assertEqual(len(gt), pred.size()[0])
    #     self.assertEqual(max(gt) + 1, pred.size()[1])
    #
    #     trainer = Trainer(self.model, self.dataloader, self.test_dataloader, self.loss_fn, self.optimizer,
    #                       checkpoints_dir=self.cp_dir, validate_every_steps=4)
    #     trainer.train(1)
    #
    #     with mock.patch('torch.load') as load:
    #         gt, pred = Trainer.test(self.test_dataloader, self.model)
    #         self.assertEqual(0, load.call_count)
    #     with mock.patch('torch.load', return_value={'state_dict': self.model.state_dict()}) as load:
    #         gt, pred = Trainer.test(self.test_dataloader, self.model, osp.join(self.cp_dir, "checkpoint_1.pth"), )
    #         self.assertEqual(1, load.call_count)
    #     self.assertEqual(len(gt), pred.size()[0])
    #     self.assertEqual(max(gt) + 1, pred.size()[1])
    #
    #     gt, pred = Trainer.test(self.test_dataloader, self.model, osp.join(self.cp_dir, "best.pth"))
    #     self.assertEqual(len(gt), pred.size()[0])
    #     self.assertEqual(max(gt) + 1, pred.size()[1])


class __constant_loss__(torch.nn.Module):
    def __init__(self, value: int = 1):
        super().__init__()
        self.value = __loss_mock__(value)


    def __call__(self, *args, **kwargs):
        return self.value


class __loss_mock__:

    def __init__(self, value: int = 1):
        self.value = value


    def backward(self):
        pass


    def item(self):
        return self.value


if __name__ == '__main__':
    unittest.main()
