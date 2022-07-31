from sacred import Ingredient
from torch import nn
from torchvision.models import DenseNet, densenet121, DenseNet121_Weights, EfficientNet, efficientnet_b0, EfficientNet_B0_Weights, resnet50, \
    efficientnet_b3, EfficientNet_B3_Weights
from typing import Tuple

from pthelpers.models.fpn import resnet50_fpn

model_builder_ingredient = Ingredient('model')


@model_builder_ingredient.capture()
def build_model(_log, model_name):
    if model_name == 'Densenet': return build_densenet()
    if model_name == 'Densenet121': return build_densenet121()
    if model_name == 'EfficientnetB0': return build_efficientnetB0()
    if model_name == 'EfficientnetB3': return build_efficientnetB3()
    if model_name == 'Resnet50': return build_resnet50()


@model_builder_ingredient.capture(prefix='densenet')
def build_densenet(_log, growth_rate: int, block_config: Tuple[int, int, int, int], num_init_features: int, bn_size: int,
                   drop_rate: float, memory_efficient, num_classes):
    densenet = DenseNet(growth_rate, block_config, num_init_features, bn_size, drop_rate, num_classes, memory_efficient)

    return nn.Sequential(
            densenet,
            nn.Sigmoid() if num_classes == 1 else nn.Softmax()
    )


@model_builder_ingredient.capture(prefix='densenet121')
def build_densenet121(_log, weights: str, frozen: bool, num_classes: int, drop_rate: float = 0):
    densenet = densenet121(weights=DenseNet121_Weights[weights] if weights else None,
                           num_classes=1000 if weights else num_classes, drop_rate=drop_rate)

    if weights and num_classes != 1000:
        num_ftrs = densenet.classifier.in_features
        densenet.classifier = nn.Linear(num_ftrs, num_classes)

    if frozen:
        for param in densenet.parameters():
            param.requires_grad = False
        # only unlock classifier
        for param in densenet.classifier.parameters():
            param.requires_grad = True

    return nn.Sequential(
            densenet,
            nn.Sigmoid() if num_classes == 1 else nn.Softmax()
    )


@model_builder_ingredient.capture(prefix='efficientnetB0')
def build_efficientnetB0(_log, weights: str = None, frozen: bool= False, num_classes: int = 1000):
    efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights[weights] if weights else None,
                                   num_classes=1000 if weights else num_classes)

    if weights and num_classes != 1000:
        num_ftrs = efficientnet.classifier[1].in_features
        efficientnet.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(num_ftrs, num_classes),
        )

    if frozen:
        for param in efficientnet.parameters():
            param.requires_grad = False
        # only unlock classifier
        for param in efficientnet.classifier.parameters():
            param.requires_grad = True

    return nn.Sequential(
            efficientnet,
            nn.Sigmoid() if num_classes == 1 else nn.Softmax()
    )

@model_builder_ingredient.capture(prefix='efficientnetB3')
def build_efficientnetB3(_log, weights: str = None, frozen: bool= False, num_classes: int = 1000):
    efficientnet = efficientnet_b3(weights=EfficientNet_B3_Weights[weights] if weights else None,
                                   num_classes=1000 if weights else num_classes)

    if weights and num_classes != 1000:
        num_ftrs = efficientnet.classifier[1].in_features
        efficientnet.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(num_ftrs, num_classes),
        )

    if frozen:
        for param in efficientnet.parameters():
            param.requires_grad = False
        # only unlock classifier
        for param in efficientnet.classifier.parameters():
            param.requires_grad = True

    return nn.Sequential(
            efficientnet,
            nn.Sigmoid() if num_classes == 1 else nn.Softmax()
    )

@model_builder_ingredient.capture(prefix='resnet50')
def build_resnet50(_log, weights: str, frozen: bool, num_classes: int, drop_rate: float):
    resnet = resnet50(weights, num_classes)

    if weights and num_classes != 1000:
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Sequential(
                nn.Linear(num_ftrs, num_classes),
        )

    if frozen:
        for param in resnet.parameters():
            param.requires_grad = False
        # only unlock classifier
        for param in resnet.fc.parameters():
            param.requires_grad = True

    return nn.Sequential(
            resnet,
            nn.Sigmoid() if num_classes == 1 else nn.Softmax()
    )
@model_builder_ingredient.capture(prefix='resnet50_fpn')
def build_resnet50_fpn(_log, weights: str, num_classes: int, drop_rate: float):
    resnet_fpn = resnet50_fpn()

    return nn.Sequential(
            resnet_fpn,
            nn.Sigmoid() if num_classes == 1 else nn.Softmax()
    )
