import numpy as np
import wandb


def get_class_names(n_classes):
    return wandb.config["class_names"] if "class_names" in wandb.config else np.arange(n_classes)