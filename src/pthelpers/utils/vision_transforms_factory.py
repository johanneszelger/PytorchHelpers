import torch
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
    if "contrast" in _config and _config["contrast"]:
        trafos.append(transforms.RandomAutocontrast(_config["contrast"]))
    if "sharpness" in _config and _config["sharpness"]:
        trafos.append(transforms.RandomAdjustSharpness(_config["sharpness"]))
    if "r_crop" in _config and _config["r_crop"]:
        trafos.append(transforms.RandomCrop((_config["r_crop"]["width"], _config["r_crop"]["height"])))

    if "noise" in _config and _config["noise"]:
        trafos.append(AddGaussianNoise(_config["noise"]["mean"], _config["noise"]["std"]))

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

    if "h_flip" in _config and _config["h_flip"]:
        trafos.append(transforms.RandomHorizontalFlip(_config["h_flip"]))
    if "v_flip" in _config and _config["v_flip"]:
        trafos.append(transforms.RandomVerticalFlip(_config["v_flip"]))

    trafos.append(transforms.ToTensor())

    if "normalize" in _config and _config["normalize"]:
        trafos.append(transforms.Normalize(_config["normalize"]["mean"], _config["normalize"]["std"]))

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


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean


    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
