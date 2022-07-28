from sacred import Ingredient
from torchvision.transforms import transforms

vision_transforms_factory_ingredient = Ingredient("transforms_factory")


# @vision_transforms_factory_ingredient.config
# def cfg():
#     resize: {
#         "width": None,
#         "height": None
#     }
#     normalize {
#         "std": None,
#         "mean": None
#     }
#     rotate = None
#     color_jitter: {
#         "brightness": None,
#         "hue": None,
#         "contrast": None,
#         "saturation": None
#     }
#     perspective: {
#         "strength": None,
#         "probability": None
#     }
#     force_3ch = False


@vision_transforms_factory_ingredient.capture
def generate_train_transforms(_config):
    trafos = []

    if "resize" in _config and _config["resize"]:
        trafos.append(transforms.Resize((_config["resize"]["width"], _config["resize"]["height"])))

    if "normalize" in _config and _config["normalize"]:
        trafos.append(transforms.Normalize(_config["std"], _config["mean"]))

    if "rotate" in _config and _config["rotate"]:
        trafos.append(transforms.RandomRotation(_config["rotate"]))

    if "color_jitter" in _config and _config["color_jitter"]:
        trafos.append(transforms.ColorJitter(
                _config["color_jitter"]["brightness"] if "brightness" in _config["color_jitter"] else 0,
                _config["color_jitter"]["contrast"] if "contrast" in _config["color_jitter"] else 0,
                _config["color_jitter"]["saturation"] if "saturation" in _config["color_jitter"] else 0,
                _config["color_jitter"]["hue"] if "hue" in _config["color_jitter"] else 0))

    if "perspective" in _config and _config["perspective"]:
        trafos.append(transforms.RandomPerspective(
                _config["perspective"]["strength"] if "strength" in _config["perspective"] else 0,
                _config["perspective"]["probability"] if "probability" in _config["perspective"] else 0))

    trafos.append(transforms.ToTensor())

    if "force_3ch" in _config and _config["force_3ch"]:
        trafos.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x))

    return transforms.Compose(trafos)


@vision_transforms_factory_ingredient.capture
def generate_test_transforms(_config):
    trafos = []

    if "resize" in _config and _config["resize"]:
        trafos.append(transforms.Resize((_config["resize"]["width"], _config["resize"]["height"])))

    trafos.append(transforms.ToTensor())

    if "force_3ch" in _config and _config["force_3ch"]:
        trafos.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x))

    return transforms.Compose(trafos)
