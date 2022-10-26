import torch
import wandb
from torchvision.transforms import transforms


class TransformsFactory:
    def __init__(self):
        if "transformation" not in wandb.config:
            raise ValueError(
                "To use the transform factory, make sure to define a transformation dict in wandb config and define all necessary params.")


    def generate_train_transforms(self) -> transforms.Compose:
        trafo_params = wandb.config["transformation"]
        trafos = []

        if "resize" in trafo_params and trafo_params["resize"]:
            trafos.append(transforms.Resize((trafo_params["resize"]["height"], trafo_params["resize"]["width"])))
        if "contrast" in trafo_params and trafo_params["contrast"] and trafo_params["contrast"] != 0:
            trafos.append(transforms.RandomAutocontrast(trafo_params["contrast"]))
        if "sharpness" in trafo_params and trafo_params["sharpness"] and trafo_params["sharpness"] != 0:
            trafos.append(transforms.RandomAdjustSharpness(trafo_params["sharpness"]))
        if "r_crop" in trafo_params and trafo_params["r_crop"] and \
                (trafo_params["r_crop"]["height"] != trafo_params["resize"]["height"] or trafo_params["r_crop"]["width"] !=
                 trafo_params["resize"]["width"]):
            trafos.append(transforms.RandomCrop((trafo_params["r_crop"]["height"], trafo_params["r_crop"]["width"])))

        if "rotate" in trafo_params and trafo_params["rotate"] and trafo_params["rotate"] != 0:
            trafos.append(transforms.RandomRotation(trafo_params["p_rotate"]))

        if "color_jitter" in trafo_params and trafo_params["color_jitter"]:
            cj = trafo_params["color_jitter"]
            if ("brightness" in cj and cj["brightness"] != 0) or \
                    ("contrast" in cj and cj["contrast"] != 0) or \
                    ("saturation" in cj and cj["saturation"] != 0) or \
                    ("hue" in cj and cj["hue"] != 0):
                trafos.append(transforms.ColorJitter(
                        cj["brightness"] if "brightness" in cj else 0,
                        cj["contrast"] if "contrast" in cj else 0,
                        cj["saturation"] if "saturation" in cj else 0,
                        cj["hue"] if "hue" in cj else 0))

        if "perspective" in trafo_params and trafo_params["perspective"]:
            per = trafo_params["perspective"]
            if ("probability" in per and per["probability"] != 0) or \
                    ("strength" in per and per["strength"] != 0):
                trafos.append(transforms.RandomPerspective(
                        per["strength"] if "strength" in per else 0,
                        per["probability"] if "probability" in per else 0))

        if "h_flip" in trafo_params and trafo_params["h_flip"] and trafo_params["h_flip"] != 0:
            trafos.append(transforms.RandomHorizontalFlip(trafo_params["h_flip"]))
        if "v_flip" in trafo_params and trafo_params["v_flip"] and trafo_params["v_flip"] != 0:
            trafos.append(transforms.RandomVerticalFlip(trafo_params["v_flip"]))

        trafos.append(transforms.ToTensor())

        if "noise" in trafo_params and trafo_params["noise"]:
            noise = trafo_params["noise"]
            if "std" in noise and noise["std"] != 0:
                trafos.append(AddGaussianNoise(noise["mean"], noise["std"]))

        if "normalize" in trafo_params and trafo_params["normalize"]:
            norm = trafo_params["normalize"]
            if ("mean" in norm and norm["mean"] != 0) or \
                    ("std" in norm and norm["std"] != 0):
                trafos.append(transforms.Normalize(norm["mean"], norm["std"]))

        if "force_3ch" in trafo_params and trafo_params["force_3ch"]:
            trafos.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x))

        return transforms.Compose(trafos)


    def generate_test_transforms(self):
        trafo_params = wandb.config["transformation"]
        trafos = []

        if "resize" in trafo_params and trafo_params["resize"]:
            trafos.append(transforms.Resize((trafo_params["resize"]["width"], trafo_params["resize"]["height"])))

        trafos.append(transforms.ToTensor())

        if "force_3ch" in trafo_params and trafo_params["force_3ch"]:
            trafos.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x))

        return transforms.Compose(trafos)


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean


    def __call__(self, tensor):
        res = tensor + torch.randn(tensor.size()) * self.std + self.mean
        res = (res - res.min()) / (res.max() - res.min())
        return res


    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
