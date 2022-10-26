import os
import random

import numpy
import numpy as np
import torch

__seed_set = None

def get_seed():
    return __seed_set

def set_seed(seed: int, deterministic:bool = False):
    '''
    Sets all seeds to make experiments reproducible
    :param seed: seed to set to
    :param deterministic: sets cuda to exact calculations
    :return: A generator and a worker creation function to put into dataloaders
    '''

    if deterministic:
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    global __seed_set
    __seed_set = seed


    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)


    g = torch.Generator()
    g.manual_seed(0)

    return g, seed_worker
