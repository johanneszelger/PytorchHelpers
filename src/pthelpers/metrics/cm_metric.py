import torch
from torch import Tensor
from torchmetrics import Metric

from pthelpers.logging.confusion_matrix import confusion_matrix


class CmMetric(Metric):
    full_state_update: bool = False

    def __init__(self, name_prefix: str, class_names):
        super().__init__()
        self.name_prefix = name_prefix
        self.class_names = class_names
        self.add_state("targets", default=torch.tensor([]), dist_reduce_fx='sum')
        self.add_state("preds", default=torch.tensor([]), dist_reduce_fx='sum')


    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets. See
        :ref:`references/modules:input types` for more information on input
        types.
        Args:
            preds: Predictions from model (logits, probabilities, or labels)
            target: Ground truth labels
        """
        one = torch.tensor(1).int()
        for i in range(len(preds)):
            had_det = False
            for j in range(len(preds[i])):
                if preds[i, j] > 0.5:
                    had_label = False
                    for k in range(len(target[i])):
                        if target[i, k] == one:
                            self.preds = torch.concat((self.preds, torch.tensor([j + 1])))
                            self.targets = torch.concat((self.targets, torch.tensor([k + 1])))
                            had_label = True
                    if not had_label:
                        self.preds = torch.concat((self.preds, torch.tensor([j + 1])))
                        self.targets = torch.concat((self.targets, torch.tensor([0])))
                    had_det = True
            if not had_det:
                had_label = False
                for k in range(len(target[i])):
                    if target[i, k] == one:
                        self.preds = torch.concat((self.preds, torch.tensor([0])))
                        self.targets = torch.concat((self.targets, torch.tensor([k + 1])))
                        had_label = True
                if not had_label:
                    self.preds = torch.concat((self.preds, torch.tensor([0])))
                    self.targets = torch.concat((self.targets, torch.tensor([0])))

    def to(self, device):
        pass
    def cuda(self, device):
        pass

    def compute(self):
        """Computes accuracy based on inputs passed in to ``update`` previously."""
        return confusion_matrix(None, self.name_prefix + "_cm_table", self.targets.numpy(), self.preds.numpy(), ["None"] + self.class_names, self.name_prefix + "_cm")


if __name__ == '__main__':
    from torchmetrics.utilities import check_forward_full_state_property

    check_forward_full_state_property(CmMetric, {"label_value": 0}, {"preds": [], "target": torch.tensor([0, 0, 1, 1, 0])})
