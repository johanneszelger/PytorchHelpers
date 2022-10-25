import torch
import wandb
from torchvision.transforms import transforms

class TransformsFactory:
    def __init__(self):
        if "transformation" not in wandb.config:
            raise ValueError("To use the transform factory, make sure to define a transformation dict in wandb config and define all necessary params.")
    
    def generate_train_transforms(self) -> transforms.Compose:
        trafo_params = wandb.config["transformation"]
        trafos = []
    
        if "resize" in trafo_params and trafo_params["resize"]:
            trafos.append(transforms.Resize((trafo_params["resize"][0], trafo_params["resize"][1])))
        if "contrast" in trafo_params and trafo_params["contrast"]:
            trafos.append(transforms.RandomAutocontrast(trafo_params["contrast"]))
        if "sharpness" in trafo_params and trafo_params["sharpness"]:
            trafos.append(transforms.RandomAdjustSharpness(trafo_params["sharpness"]))
        if "r_crop" in trafo_params and trafo_params["r_crop"]:
            trafos.append(transforms.RandomCrop((trafo_params["r_crop"][0], trafo_params["r_crop"][1])))
    
        if "rotate" in trafo_params and trafo_params["rotate"]:
            trafos.append(transforms.RandomRotation(trafo_params["rotate"]))
    
        if "color_jitter" in trafo_params and trafo_params["color_jitter"]:
            trafos.append(transforms.ColorJitter(
                    trafo_params["color_jitter"]["brightness"] if "brightness" in trafo_params["color_jitter"] else 0,
                    trafo_params["color_jitter"]["contrast"] if "contrast" in trafo_params["color_jitter"] else 0,
                    trafo_params["color_jitter"]["saturation"] if "saturation" in trafo_params["color_jitter"] else 0,
                    trafo_params["color_jitter"]["hue"] if "hue" in trafo_params["color_jitter"] else 0))
    
        if "perspective" in trafo_params and trafo_params["perspective"]:
            trafos.append(transforms.RandomPerspective(
                    trafo_params["perspective"]["strength"] if "strength" in trafo_params["perspective"] else 0,
                    trafo_params["perspective"]["probability"] if "probability" in trafo_params["perspective"] else 0))
    
        if "h_flip" in trafo_params and trafo_params["h_flip"]:
            trafos.append(transforms.RandomHorizontalFlip(trafo_params["h_flip"]))
        if "v_flip" in trafo_params and trafo_params["v_flip"]:
            trafos.append(transforms.RandomVerticalFlip(trafo_params["v_flip"]))
    
        trafos.append(transforms.ToTensor())
    
        if "noise" in trafo_params and trafo_params["noise"]:
            trafos.append(AddGaussianNoise(trafo_params["noise"]["mean"], trafo_params["noise"]["std"]))
    
        if "normalize" in trafo_params and trafo_params["normalize"]:
            trafos.append(transforms.Normalize(trafo_params["normalize"]["mean"], trafo_params["normalize"]["std"]))
    
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
