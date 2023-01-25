import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor
from torchmetrics import Metric, Accuracy

class LabelCount(Metric):
    full_state_update: bool = False


    def __init__(self, label_value: int):
        super().__init__()
        self.label_value = label_value
        self.add_state("count", default=torch.Tensor(), dist_reduce_fx='mean')
        self.panel = "data"


    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets. See
        :ref:`references/modules:input types` for more information on input
        types.
        Args:
            preds: Predictions from model (logits, probabilities, or labels)
            target: Ground truth labels
        """
        self.append(roc_auc_score(target[:, self.label_value], preds[:, self.label_value]))


    def compute(self) -> Tensor:
        """Computes accuracy based on inputs passed in to ``update`` previously."""
        return torch.mean(self.count)


if __name__ == '__main__':
    from torchmetrics.utilities import check_forward_full_state_property

    check_forward_full_state_property(LabelCount, {"label_value": 0}, {"preds": [], "target": torch.tensor([0, 0, 1, 1, 0])})
