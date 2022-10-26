import unittest

import torch

from src.pthelpers.metrics.label_count import LabelCount


class LabelCountMetric(unittest.TestCase):
    def setUp(self) -> None:
        self.metric0 = LabelCount(0)
        self.metric1 = LabelCount(1)
        self.metric2 = LabelCount(2)
        self.metric3 = LabelCount(3)
        self.metric4 = LabelCount(4)


    def tearDown(self) -> None:
        pass


    def update(self, tensor):
        self.metric0.update(torch.Tensor(), tensor)
        self.metric1.update(torch.Tensor(), tensor)
        self.metric2.update(torch.Tensor(), tensor)
        self.metric3.update(torch.Tensor(), tensor)
        self.metric4.update(torch.Tensor(), tensor)


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
