import unittest

from pthelpers.models.simple_net import SimpleNet
from pthelpers.training.trainer import Trainer
from test.models.model_test import ModelTest


class TestSimpleNet(ModelTest):
    def __init__(self, methodName):
        super().__init__(methodName)

    def test_simplenet_training(self):
        self.model = SimpleNet()
        Trainer(self.train_loader, self.test_loader, 10).train(self.model, self.optimizer, 5, self.scheduler)
        self._assert_cp_count()

if __name__ == '__main__':
    unittest.main()