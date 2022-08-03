import torch
from torch import Tensor
from torchmetrics import Metric, Accuracy

Accuracy()
class LabelCount(Metric):
    full_state_update: bool = False


    def __init__(self, label_value: int):
        super().__init__()
        self.label_value = label_value
        self.add_state("count", default=torch.zeros(1), dist_reduce_fx='sum')


    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets. See
        :ref:`references/modules:input types` for more information on input
        types.
        Args:
            preds: Predictions from model (logits, probabilities, or labels)
            target: Ground truth labels
        """
        if target.dim() == 1:
            bincounts = torch.bincount(target)
            if len(bincounts) - 1 < self.label_value:
                return
            self.count += bincounts[self.label_value]
        elif target.dim() == 2:
            for row in target:
                if len(row) - 1 < self.label_value:
                    return
                self.count += row[self.label_value]
        else:
            raise ValueError("unknown dimension of targets, use 1D for numbers and 2D for categorical data")



    def compute(self) -> Tensor:
        """Computes accuracy based on inputs passed in to ``update`` previously."""
        return self.count


if __name__ == '__main__':
    from torchmetrics.utilities import check_forward_full_state_property

    check_forward_full_state_property(LabelCount, {"label_value": 0}, {"preds": [], "target": torch.tensor([0, 0, 1, 1, 0])})
