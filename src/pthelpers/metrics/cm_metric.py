import torch
from torch import Tensor
from torchmetrics import Metric

from src.pthelpers.logging.confusion_matrix import confusion_matrix


class CmMetric(Metric):
    full_state_update: bool = False

    def __init__(self, name_prefix: str, class_names, label_value: int):
        super().__init__()
        self.name_prefix = name_prefix
        self.class_names = class_names
        self.label_value = label_value
        self.add_state("targets", default=torch.zeros(1), dist_reduce_fx='sum')
        self.add_state("preds", default=torch.zeros(1), dist_reduce_fx='sum')


    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets. See
        :ref:`references/modules:input types` for more information on input
        types.
        Args:
            preds: Predictions from model (logits, probabilities, or labels)
            target: Ground truth labels
        """
        if target.ndim == 1:
            self.targets += (target == self.label_value).sum()
        else:
            self.targets += target[:, self.label_value].sum()
        if preds.ndim == 1:
            self.preds += (target == self.label_value).sum()
        else:
            self.preds += target[:, self.label_value].sum()


    def compute(self) -> Tensor:
        """Computes accuracy based on inputs passed in to ``update`` previously."""
        confusion_matrix(None, self.name + "_cm_table" + self.name, self.targets, self.preds, self.classnames, self.name_prefix + "_cm")
        return None


if __name__ == '__main__':
    from torchmetrics.utilities import check_forward_full_state_property

    check_forward_full_state_property(CmMetric, {"label_value": 0}, {"preds": [], "target": torch.tensor([0, 0, 1, 1, 0])})
