import os

import wandb

from test.mnist_test import MnistTest

class ModelTest(MnistTest):
    def setUp(self) -> None:
        super(ModelTest, self).setUp()
        wandb.config.update({"training": {"cp_base_path": "checkpoints", "log_interval_batches": 5, "save_every_nth_epoch": 5,
                                          "plot_class_dist": False, "plot_samples": False}},
                            allow_val_change=True)

    def _assert_cp_count(self, count: int = 2) -> None:
        cp_dir = os.path.join("checkpoints", wandb.run.name)
        files = os.listdir(cp_dir)
        assert len(files) == count, f"Expected {count} checkpoints, but got {len(files)}"
