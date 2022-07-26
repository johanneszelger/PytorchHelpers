from sacred import Ingredient
from torchvision.transforms import transforms

vision_transforms_factory_ingredient = Ingredient("transforms_factory")


@vision_transforms_factory_ingredient.config
def cfg():
    resize: {
        "width": None,
        "height": None
    }
    rotate = None
    color_jitter: {
        "brightness": None,
        "hue": None,
        "contrast": None,
        "saturation": None
    }
    perspective: {
        "strength": None,
        "probability": None
    }
    force_3ch = False


@vision_transforms_factory_ingredient.capture
def generate_train_transforms(_config):
    trafos = []
    if _config["resize"]:
        trafos.append(transforms.Resize((_config["resize"]["width"], _config["resize"]["height"])))

    if _config["rotate"]:
        trafos.append(transforms.RandomRotation(_config["rotate"]))

    if _config["color_jitter"]:
        trafos.append(transforms.ColorJitter(_config["color_jitter"]["brightness"], _config["color_jitter"]["contrast"],
                                             _config["color_jitter"]["saturation"], _config["color_jitter"]["hue"]))

    if _config["perspective"]:
        trafos.append(transforms.Resize((_config["resize"]["width"], _config["resize"]["height"])))

    trafos.append(transforms.ToTensor())

    if _config["force_3ch"]:
        trafos.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x))

    return transforms.Compose(trafos)


@vision_transforms_factory_ingredient.capture
def generate_test_transforms(_config):
    trafos = []
    if _config["width"] and _config["height"]:
        trafos.append(transforms.Resize((_config["width"], _config["height"])))

    trafos.append(transforms.ToTensor())

    if _config["force_3ch"]:
        trafos.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x))

    return transforms.Compose(trafos)
