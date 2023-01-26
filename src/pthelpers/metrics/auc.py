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


    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets. See
        :ref:`references/modules:input types` for more information on input
        types.
        Args:
            preds: Predictions from model (logits, probabilities, or labels)
            target: Ground truth labels
        """
        self.targets=torch.cat([self.targets, target[:, self.label_value].cpu()])
        self.preds=torch.cat([self.preds, preds[:, self.label_value].cpu()])

    def to(self, device):
        pass
    def compute(self) -> Tensor:
        """Computes accuracy based on inputs passed in to ``update`` previously."""
        try:
            return roc_auc_score(self.targets, self.preds)
        except:
            return 0

if __name__ == '__main__':
    from torchmetrics.utilities import check_forward_full_state_property

    check_forward_full_state_property(AUC, {"label_value": 0}, {"preds": torch.tensor([[0.25, 0.12, 0.11, 0.8]]),
                                                                "target": torch.tensor([[0, 0, 0, 1]])})
