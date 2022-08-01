import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from torch.nn import Module
from torch.utils.data import DataLoader

from src.pthelpers.training.trainer import Trainer


def plot_auc(data_loader: DataLoader, class_names: [], checkpoint: str = None, model: Module = None,
             use_gpu: bool = True, show: bool = True, title: str = None):
    """
    allows plotting roc.py curves of a model on a given dataloader
    :param data_loader: dataloader to test on
    :param class_names: names of the classes in the dataset
    :param checkpoint: path to a trainer/model checkpoint
    :param model: a model to evaluate
    :param use_gpu: if gpu acceleration should be used
    :param show: show the plot
    :return:
    """

    gt, pred = Trainer.test(model, checkpoint, data_loader, use_gpu)
    gt, pred = gt.int().cpu().numpy(), pred.cpu().numpy()
    if gt.ndim < 2:
        gt = np.eye(pred.shape[1], dtype='uint8')[gt]

    if pred.shape[1] != len(class_names):
        raise ValueError(f"there are {pred.shape[1]} classes, but only {len(class_names)} where given!")

    for i in range(pred.shape[1]):
        fpr, tpr, threshold = metrics.roc_curve(gt[:, i], pred[:, i])
        roc_auc = metrics.auc(fpr, tpr)

        plt.title('ROC for: ' + str(class_names[i]) if title is None else title)
        plt.plot(fpr, tpr, label=f'{class_names[i]}: AUC = %0.2f' % roc_auc)

        plt.legend(loc='lower right')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')

        if show:
            plt.show()