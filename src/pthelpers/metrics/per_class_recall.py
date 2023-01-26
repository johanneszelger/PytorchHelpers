import torch
from torch import Tensor
from torchmetrics import Metric, Accuracy
from torchmetrics.classification import BinaryAccuracy, BinaryRecall


class PerClassRecall(Metric):
    full_state_update: bool = False
    def __init__(self, class_idx):
        super().__init__()
        self.class_idx = class_idx
        self.rec = BinaryRecall(multidim_average='global')


    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets. See
        :ref:`references/modules:input types` for more information on input
        types.
        Args:
            preds: Predictions from model (logits, probabilities, or labels)
            target: Ground truth labels
        """
        self.rec.update(preds[:, self.class_idx], target[:, self.class_idx])


    def compute(self) -> Tensor:
        """Computes accuracy based on inputs passed in to ``update`` previously."""
        return self.rec.compute()

    def reset(self) -> None:
        self.rec.reset()

