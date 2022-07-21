from sacred import Ingredient
from torchvision.transforms import transforms

vision_transformation_factory_ingredient = Ingredient("transformation_factory")

@vision_transformation_factory_ingredient.config
def cfg():
    width: None
    height: None
    force3Ch: False

def generate_train_transformer(_config):
    trafos = []
    if _config["width"] and _config["height"]:
        trafos.append(transforms.Resize((_config["width"], _config["height"])))

    trafos.append(transforms.ToTensor())

    if _config["force3Ch"]:
        trafos.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x))

    return transforms.Compose(trafos)

def generate_test_transformer(_config):
    trafos = []
    if _config["width"] and _config["height"]:
        trafos.append(transforms.Resize((_config["width"], _config["height"])))

    trafos.append(transforms.ToTensor())

    if _config["force3Ch"]:
        trafos.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x))

    return transforms.Compose(trafos)