import unittest

import torch

from pthelpers.metrics.prediction_count import PredictionCount


class LabelCountMetric(unittest.TestCase):
    def setUp(self) -> None:
        self.metric0 = PredictionCount(0)
        self.metric1 = PredictionCount(1)
        self.metric2 = PredictionCount(2)
        self.metric3 = PredictionCount(3)
        self.metric4 = PredictionCount(4)


    def tearDown(self) -> None:
        pass


    def update(self, tensor):
        self.metric0.update(tensor, torch.Tensor())
        self.metric1.update(tensor, torch.Tensor())
        self.metric2.update(tensor, torch.Tensor())
        self.metric3.update(tensor, torch.Tensor())
        self.metric4.update(tensor, torch.Tensor())


    def test_update(self):
        self.update(torch.tensor([0, 2, 2, 3, 3, 3]))

        self.assertEqual(1, self.metric0.compute().item())
        self.assertEqual(0, self.metric1.compute().item())
        self.assertEqual(2, self.metric2.compute().item())
        self.assertEqual(3, self.metric3.compute().item())
        self.assertEqual(0, self.metric4.compute().item())

        self.update(torch.tensor([2, 2, 2, 2, 3, 3, 4]))

        self.assertEqual(1, self.metric0.compute().item())
        self.assertEqual(0, self.metric1.compute().item())
        self.assertEqual(6, self.metric2.compute().item())
        self.assertEqual(5, self.metric3.compute().item())
        self.assertEqual(1, self.metric4.compute().item())


if __name__ == '__main__':
    unittest.main()
