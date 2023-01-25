import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch import Tensor
from torchmetrics import Metric, Accuracy

class AUC(Metric):
    full_state_update: bool = False


    def __init__(self, label_value: int):
        super().__init__()
        self.label_value = label_value
        self.add_state("targets", default=torch.Tensor(), dist_reduce_fx='mean')
        self.add_state("preds", default=torch.Tensor(), dist_reduce_fx='mean')
        self.panel = "data"


    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets. See
        :ref:`references/modules:input types` for more information on input
        types.
        Args:
            preds: Predictions from model (logits, probabilities, or labels)
            target: Ground truth labels
        """
        self.target.append(target[:, self.label_value])
        self.preds.append(preds[:, self.label_value])


    def compute(self) -> Tensor:
        """Computes accuracy based on inputs passed in to ``update`` previously."""
        return roc_auc_score(self.targets, self.preds)


if __name__ == '__main__':
    from torchmetrics.utilities import check_forward_full_state_property

    check_forward_full_state_property(AUC, {"label_value": 0}, {"preds": torch.tensor([[0.25, 0.12, 0.11, 0.8]]),
                                                                "target": torch.tensor([[0, 0, 0, 1]])})
