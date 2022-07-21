import random

import numpy as np
import torch


class Reproducer:
    seed_set = False


    @staticmethod
    def set_seed(seed: int):
        '''
        Sets all seeds to make experiments reproducible
        :param seed: seed to set to
        :return: A generator and a worker creation function to put into dataloaders
        '''
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        seed_set = True


        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2 ** 32


        g = torch.Generator()
        g.manual_seed(0)

        return g, seed_worker
