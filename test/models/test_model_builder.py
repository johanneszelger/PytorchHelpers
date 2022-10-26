import unittest

import wandb

from src.pthelpers.models import ModelBuilder
from src.pthelpers.training import Trainer
from test.models.model_test import ModelTest


class TestModelBuilder(ModelTest):
    def __init__(self, methodName):
        super().__init__(methodName, size=(128, 128), rgb=True, batch_size=50)


    def setUp(self) -> None:
        super(TestModelBuilder, self).setUp()
        wandb.config.update({"model": {}})


    def tearDown(self) -> None:
        self._assert_cp_count(2)
        super(TestModelBuilder, self).tearDown()


    def test_densenet_training(self):
        model_params = wandb.config["model"]
        model_params["name"] = "DenseNet"
        model_params["growth_rate"] = 32
        model_params["block_config"] = (6, 12, 24, 16)
        model_params["num_init_features"] = 64
        model_params["bn_size"] = 4
        model_params["drop_rate"] = 0
        model_params["num_classes"] = 1000
        model_params["memory_efficient"] = False

        self.model = ModelBuilder().build_model()
        assert model_params["name"] == self.model[0].__class__.__name__, \
            f"Expected model name {model_params['name']}, got {self.model[0].__class__.__name__}"
        Trainer(self.train_loader, self.test_loader).train(self.model, self.optimizer, 5, self.scheduler)


    def test_densenet121_training(self):
        model_params = wandb.config["model"]
        model_params["name"] = "DenseNet121"

        self.model = ModelBuilder().build_model()
        assert model_params["name"].startswith(self.model[0].__class__.__name__), \
            f"Expected model name {model_params['name']}, got {self.model[0].__class__.__name__}"
        Trainer(self.train_loader, self.test_loader).train(self.model, self.optimizer, 5, self.scheduler)


    def test_efficientnetB0_training(self):
        model_params = wandb.config["model"]
        model_params["name"] = "EfficientNetB0"
        self.model = ModelBuilder().build_model()
        assert model_params["name"].startswith(self.model[0].__class__.__name__), \
            f"Expected model name {model_params['name']}, got {self.model[0].__class__.__name__}"
        Trainer(self.train_loader, self.test_loader).train(self.model, self.optimizer, 5, self.scheduler)


    def test_efficientnetB1_training(self):
        model_params = wandb.config["model"]
        model_params["name"] = "EfficientNetB1"
        self.model = ModelBuilder().build_model()
        assert model_params["name"].startswith(self.model[0].__class__.__name__), \
            f"Expected model name {model_params['name']}, got {self.model[0].__class__.__name__}"
        Trainer(self.train_loader, self.test_loader).train(self.model, self.optimizer, 5, self.scheduler)


    def test_efficientnetB2_training(self):
        model_params = wandb.config["model"]
        model_params["name"] = "EfficientNetB2"
        self.model = ModelBuilder().build_model()
        assert model_params["name"].startswith(self.model[0].__class__.__name__), \
            f"Expected model name {model_params['name']}, got {self.model[0].__class__.__name__}"
        Trainer(self.train_loader, self.test_loader).train(self.model, self.optimizer, 5, self.scheduler)


    def test_efficientnetB3_training(self):
        model_params = wandb.config["model"]
        model_params["name"] = "EfficientNetB3"
        self.model = ModelBuilder().build_model()
        assert model_params["name"].startswith(self.model[0].__class__.__name__), \
            f"Expected model name {model_params['name']}, got {self.model[0].__class__.__name__}"
        Trainer(self.train_loader, self.test_loader).train(self.model, self.optimizer, 5, self.scheduler)

    def test_resnet18_training(self):
        model_params = wandb.config["model"]
        model_params["name"] = "ResNet18"
        self.model = ModelBuilder().build_model()
        assert model_params["name"].startswith(self.model[0].__class__.__name__), \
            f"Expected model name {model_params['name']}, got {self.model[0].__class__.__name__}"
        Trainer(self.train_loader, self.test_loader).train(self.model, self.optimizer, 5, self.scheduler)

    def test_resnet34_training(self):
        model_params = wandb.config["model"]
        model_params["name"] = "ResNet34"
        self.model = ModelBuilder().build_model()
        assert model_params["name"].startswith(self.model[0].__class__.__name__), \
            f"Expected model name {model_params['name']}, got {self.model[0].__class__.__name__}"
        Trainer(self.train_loader, self.test_loader).train(self.model, self.optimizer, 5, self.scheduler)

    def test_resnet50_training(self):
        model_params = wandb.config["model"]
        model_params["name"] = "ResNet50"
        self.model = ModelBuilder().build_model()
        assert model_params["name"].startswith(self.model[0].__class__.__name__), \
            f"Expected model name {model_params['name']}, got {self.model[0].__class__.__name__}"
        Trainer(self.train_loader, self.test_loader).train(self.model, self.optimizer, 5, self.scheduler)

    def test_resnet101_training(self):
        model_params = wandb.config["model"]
        model_params["name"] = "ResNet101"
        self.model = ModelBuilder().build_model()
        assert model_params["name"].startswith(self.model[0].__class__.__name__), \
            f"Expected model name {model_params['name']}, got {self.model[0].__class__.__name__}"
        Trainer(self.train_loader, self.test_loader).train(self.model, self.optimizer, 5, self.scheduler)

    def test_resnet152_training(self):
        model_params = wandb.config["model"]
        model_params["name"] = "ResNet152"
        self.model = ModelBuilder().build_model()
        assert model_params["name"].startswith(self.model[0].__class__.__name__), \
            f"Expected model name {model_params['name']}, got {self.model[0].__class__.__name__}"
        Trainer(self.train_loader, self.test_loader).train(self.model, self.optimizer, 5, self.scheduler)


if __name__ == '__main__':
    unittest.main()
