import numpy as np
from sklearn import metrics
from torch.utils.data import DataLoader

try:
    from pthelpers.training.trainer import Trainer
except:
    from src.pthelpers.training.trainer import Trainer


def calc_auc(data_loader: DataLoader, model, checkpoint: str, use_gpu: bool = True):
    """
    allows plotting roc.py curves of a model on a given dataloader
    :param data_loader: dataloader to test on
    :param checkpoint: path to a trainer/model checkpoint
    :param model: a model to evaluate
    :param use_gpu: if gpu acceleration should be used
    :return:
    """

    gt, pred = Trainer.test_static(model, checkpoint, data_loader, use_gpu)
    gt, pred = gt.int().cpu().numpy(), pred.cpu().numpy()
    if gt.ndim < 2:
        gt = np.eye(pred.shape[1], dtype='uint8')[gt]

    roc_aucs = []
    for i in range(pred.shape[1]):
        fpr, tpr, threshold = metrics.roc_curve(gt[:, i], pred[:, i])
        roc_aucs.append(metrics.auc(fpr, tpr))

    return roc_aucs
