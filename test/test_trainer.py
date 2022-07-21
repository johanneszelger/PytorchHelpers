import os
import os.path as osp
import shutil
import time
import unittest

import torch.optim
import torchvision.models
from pthelpers.trainer import Trainer
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from src.pthelpers.models.simple_net import Net


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

        self.log_dir = 'logs'
        if osp.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
        os.makedirs(self.log_dir)


    def tearDown(self) -> None:
        time.sleep(0.3)
        shutil.rmtree(self.cp_dir)
        shutil.rmtree(self.log_dir)


    def test_init(self):
        trainer = Trainer(self.model, self.dataloader, self.test_dataloader, self.loss_fn, self.optimizer)
        self.assertIsNotNone(trainer)


    # def test_train(self):
    #     self.optimizer.step = MagicMock(return_value=None)
    #     trainer = Trainer(self.model, self.dataloader, self.test_dataloader, self.loss_fn, self.optimizer)
    #     trainer.train(1)
    #     self.optimizer.step.assert_called()
    #
    #
    # def test_logging(self):
    #     with mock.patch('builtins.input', return_value="no"):
    #         trainer = Trainer(self.model, self.dataloader, self.test_dataloader, self.loss_fn, self.optimizer,
    #                           log_dir=self.log_dir, validate_every_steps=4, use_gpu=False)
    #         with mock.patch.object(SummaryWriter, 'add_scalar') as writer:
    #             trainer.train(1)
    #             # LR and Loss, vloss after 4 steps
    #             self.assertEqual(3, writer.call_count)
    #             self.assertEqual(2, len(os.listdir(self.log_dir)))
    #             self.assertEqual(1, len(os.listdir(osp.join(self.log_dir, "train"))))
    #
    #         trainer = Trainer(self.model, self.dataloader, self.test_dataloader, self.loss_fn, self.optimizer, use_gpu=False,
    #                           log_dir=self.log_dir, validate_every_steps=3)
    #         with mock.patch.object(SummaryWriter, 'add_scalar') as writer:
    #             trainer.train(1)
    #             # after 3 steps (loss, lr, vloss) and once at end of epoch
    #             self.assertEqual(3 + 3, writer.call_count)
    #             self.assertEqual(2, len(os.listdir(self.log_dir)))
    #             # now two files, one of the first run, one of the second run!
    #             self.assertEqual(2, len(os.listdir(osp.join(self.log_dir, "train"))))
    #
    #     with mock.patch('builtins.input', return_value="yes"):
    #         trainer = Trainer(self.model, self.dataloader, self.test_dataloader, self.loss_fn, self.optimizer,
    #                           log_dir=self.log_dir, validate_every_steps=3, use_gpu=False)
    #         trainer.train(1)
    #         trainer = Trainer(self.model, self.dataloader, self.test_dataloader, self.loss_fn, self.optimizer, use_gpu=False,
    #                           log_dir=self.log_dir, validate_every_steps=3)
    #         trainer.train(1)
    #         # now only one file, as the first run gets deleted, one of the second run!
    #         self.assertEqual(1, len(os.listdir(osp.join(self.log_dir, "train"))))
    #
    #
    # def test_evaluate_every(self):
    #     with mock.patch('builtins.input', return_value="no"):
    #         trainer = Trainer(self.model, self.dataloader, self.test_dataloader, self.loss_fn, self.optimizer,
    #                           validate_every_steps=1,
    #                           log_dir=self.log_dir, checkpoints_dir=self.cp_dir)
    #         trainer.__validate__ = MagicMock(return_value=None)
    #         with mock.patch('torch.save') as save:
    #             with mock.patch.object(SummaryWriter, 'add_scalar') as writer:
    #                 trainer.train(1)
    #
    #                 # 4 times (for every step in the epoch)
    #                 self.assertEqual(4, trainer.__validate__.call_count)
    #                 # 4 for loss 4 for lr
    #                 self.assertEqual(8, writer.call_count)
    #                 self.assertEqual(1, save.call_count)
    #
    #         trainer = Trainer(self.model, self.dataloader, self.test_dataloader, self.loss_fn, self.optimizer,
    #                           log_dir=self.log_dir, validate_every_steps=4)
    #         trainer.__validate__ = MagicMock(return_value=None)
    #         trainer.train(1)
    #         # for val once
    #         self.assertEqual(1, trainer.__validate__.call_count)
    #
    #
    # def test_evaluation(self):
    #     with mock.patch('builtins.input', return_value="no"):
    #         trainer = Trainer(self.model, self.dataloader, self.test_dataloader, self.loss_fn, self.optimizer,
    #                           log_dir=self.log_dir, eval_first=False, validate_every_steps=4)
    #
    #         with mock.patch.object(SummaryWriter, 'add_scalar') as writer:
    #             trainer.train(1)
    #             # LR and Loss Loss_v at end of epoch
    #             self.assertEqual(3, writer.call_count)
    #
    #         trainer = Trainer(self.model, self.dataloader, self.test_dataloader, self.loss_fn, self.optimizer, eval_first=True,
    #                           log_dir=self.log_dir, validate_every_steps=4)
    #         trainer.__calc_metrics_print_save__ = MagicMock(return_value=None)
    #         trainer.train(1)
    #
    #         call_args_train_start = trainer.__calc_metrics_print_save__.call_args_list[0][0]
    #         call_args_test_start = trainer.__calc_metrics_print_save__.call_args_list[1][0]
    #         call_args_train_4steps = trainer.__calc_metrics_print_save__.call_args_list[2][0]
    #         call_args_test_epoch = trainer.__calc_metrics_print_save__.call_args_list[3][0]
    #
    #         # both tensors as long as dataset
    #         self.assertEqual(len(self.data), len(call_args_train_start[0]))
    #         self.assertEqual(len(self.data), len(call_args_train_start[1]))
    #         self.assertEqual(len(self.test_data), len(call_args_test_start[0]))
    #         self.assertEqual(len(self.test_data), len(call_args_test_start[1]))
    #         self.assertEqual(len(self.data), len(call_args_train_4steps[0]))
    #         self.assertEqual(len(self.data), len(call_args_train_4steps[1]))
    #         self.assertEqual(len(self.test_data), len(call_args_test_epoch[0]))
    #         self.assertEqual(len(self.test_data), len(call_args_test_epoch[1]))
    #
    #         # step is 4 (one epoch has gone, 4 batches)
    #         self.assertEqual(0, call_args_train_start[2])
    #         self.assertEqual(0, call_args_test_start[2])
    #         self.assertEqual(4, call_args_train_4steps[2])
    #         self.assertEqual(4, call_args_test_epoch[2])
    #
    #         # validation is true
    #         self.assertEqual(False, call_args_train_start[3])
    #         self.assertEqual(True, call_args_test_start[3])
    #         self.assertEqual(False, call_args_train_4steps[3])
    #         self.assertEqual(True, call_args_test_epoch[3])
    #
    #
    # def test_metrics_print(self):
    #     with mock.patch('builtins.input', return_value="no"):
    #         trainer = Trainer(self.model, self.dataloader, self.test_dataloader, self.loss_fn, self.optimizer,
    #                           checkpoints_dir=self.cp_dir, validate_every_steps=4,
    #                           log_dir=self.log_dir)
    #         with mock.patch.object(SummaryWriter, 'add_scalar') as writer:
    #             with mock.patch('torch.save') as save:
    #                 # make sure model does not improve!
    #                 self.optimizer.step = MagicMock()
    #                 trainer.train(1)
    #
    #                 # once per epoch and once because best validation
    #                 self.assertEqual(2, save.call_count)
    #
    #                 # once per epoch, but not again for validation
    #                 trainer.__best_validation_loss = 0
    #                 trainer.train(2)
    #                 self.assertEqual(3, save.call_count)
    #
    #                 # twice for train (loss, lr) once for val (loss) at the end of epoch, but twice because 2 epochs
    #                 self.assertEqual(3 * 2, writer.call_count)
    #
    #                 trainer.metrics = {'acc': torchmetrics.Accuracy().to("cuda" if torch.cuda.is_available() else "cpu")}
    #                 trainer.train(3)
    #                 # +1 for loss and +1 for acc times 2 for train/test and +1 for train lr
    #                 self.assertEqual(3 * 2 + 5, trainer.writer_train.add_scalar.call_count)
    #
    #
    # def test_tensor_metrics(self):
    #     with mock.patch('builtins.input', return_value="no"):
    #         trainer = Trainer(self.model, self.dataloader, self.test_dataloader, self.loss_fn, self.optimizer,
    #                           log_dir=self.log_dir, validate_every_steps=4)
    #         with mock.patch.object(SummaryWriter, 'add_scalar') as add_scalar:
    #             trainer.metrics = {'acc1': torchmetrics.Accuracy().to("cuda" if torch.cuda.is_available() else "cpu"),
    #                                'acc2': torchmetrics.Accuracy(average='none', num_classes=10)
    #                                    .to("cuda" if torch.cuda.is_available() else "cpu")}
    #             trainer.train(1)
    #             # LR, Loss and ACC1 for train and test (5)
    #             # plus for every class acc train and test (20)
    #             self.assertEqual(25, add_scalar.call_count)
    #
    #
    # def test_pr_curves(self):
    #     with mock.patch('builtins.input', return_value="no"):
    #         trainer = Trainer(self.model, self.dataloader, self.test_dataloader, self.loss_fn, self.optimizer,
    #                           log_dir=self.log_dir, validate_every_steps=4)
    #         with mock.patch.object(SummaryWriter, 'add_pr_curve') as add_pr_curve:
    #             trainer.train(1)
    #             self.assertEqual(10, add_pr_curve.call_count)
    #
    #
    # def test_checkpoint_creation(self):
    #     trainer = Trainer(self.model, self.dataloader, self.test_dataloader, self.loss_fn, self.optimizer,
    #                       checkpoints_dir=self.cp_dir, validate_every_steps=4,
    #                       remove_cp_after_training=False)
    #     trainer.train(3)
    #
    #     self.assertTrue(osp.exists(osp.join(self.cp_dir, "checkpoint_1.pth")))
    #     self.assertTrue(osp.exists(osp.join(self.cp_dir, "checkpoint_2.pth")))
    #     self.assertTrue(osp.exists(osp.join(self.cp_dir, "checkpoint_3.pth")))
    #     self.assertTrue(osp.exists(osp.join(self.cp_dir, "best.pth")))
    #
    #
    # def test_checkpoint_cleanup(self):
    #     trainer = Trainer(self.model, self.dataloader, self.test_dataloader, self.loss_fn, self.optimizer,
    #                       checkpoints_dir=self.cp_dir, validate_every_steps=4,
    #                       remove_cp_after_training=True)
    #     trainer.train(3)
    #
    #     self.assertFalse(osp.exists(osp.join(self.cp_dir, "checkpoint_1.pth")))
    #     self.assertFalse(osp.exists(osp.join(self.cp_dir, "checkpoint_2.pth")))
    #     self.assertTrue(osp.exists(osp.join(self.cp_dir, "checkpoint_3.pth")))
    #     self.assertTrue(osp.exists(osp.join(self.cp_dir, "best.pth")))
    #
    #
    # def test_save_and_continue(self):
    #     trainer = Trainer(self.model, self.dataloader, self.test_dataloader, __constant_loss__(), self.optimizer,
    #                       checkpoints_dir=self.cp_dir, validate_every_steps=4)
    #     trainer.train(1)
    #
    #     trainer = Trainer(self.model, self.dataloader, self.test_dataloader, __constant_loss__(), self.optimizer,
    #                       checkpoints_dir=self.cp_dir, validate_every_steps=4)
    #     trainer.load(os.path.join(self.cp_dir, "checkpoint_1.pth"))
    #     trainer.loss_fn = __constant_loss__(2)
    #     with mock.patch('torch.save') as save:
    #         trainer.train(2)
    #         self.assertEqual(1, save.call_count)
    #         self.assertEqual(os.path.join(self.cp_dir, "checkpoint_2.pth"), save.call_args[0][1])
    #
    #     trainer.loss_fn = __constant_loss__(0)
    #     with mock.patch('torch.save') as save:
    #         trainer.train(3)
    #         self.assertEqual(2, save.call_count)
    #         self.assertEqual(os.path.join(self.cp_dir, "best.pth"), save.call_args_list[0][0][1])
    #         self.assertEqual(os.path.join(self.cp_dir, "checkpoint_3.pth"), save.call_args_list[1][0][1])
    #
    #
    # def test_save_and_auto_load(self):
    #     with mock.patch('builtins.input', return_value="yes"):
    #         trainer = Trainer(self.model, self.dataloader, self.test_dataloader, self.loss_fn, self.optimizer,
    #                           validate_every_steps=2,
    #                           checkpoints_dir=self.cp_dir, load_latest_existing=True, log_dir=self.log_dir)
    #         trainer.train(2)
    #         trainer = Trainer(self.model, self.dataloader, self.test_dataloader, self.loss_fn, self.optimizer,
    #                           checkpoints_dir=self.cp_dir, load_latest_existing=True)
    #         self.assertEqual(2, trainer.current_epoch)
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


class __constant_loss__:
    def __init__(self, value: int = 1):
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
