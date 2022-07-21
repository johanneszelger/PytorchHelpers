from sacred import Ingredient
from torchvision.transforms import transforms

vision_transforms_factory_ingredient = Ingredient("data.transforms_factory")


@vision_transforms_factory_ingredient.config
def cfg():
    width = None
    height = None
    force_3ch = False


@vision_transforms_factory_ingredient.capture
def generate_train_transforms(_config):
    trafos = []
    if _config["width"] and _config["height"]:
        trafos.append(transforms.Resize((_config["width"], _config["height"])))

    trafos.append(transforms.ToTensor())

    if _config["force3Ch"]:
        trafos.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x))

    return transforms.Compose(trafos)


@vision_transforms_factory_ingredient.capture
def generate_test_transforms(_config):
    trafos = []
    if _config["width"] and _config["height"]:
        trafos.append(transforms.Resize((_config["width"], _config["height"])))

    trafos.append(transforms.ToTensor())

    if _config["force3Ch"]:
        trafos.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x))

    return transforms.Compose(trafos)
