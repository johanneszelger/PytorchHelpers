from sacred import Ingredient
from torch import nn
from torchvision.models import DenseNet, densenet121

model_builder_ingredient = Ingredient('model')

@model_builder_ingredient.capture()
def build_model(_log, model_name):
    if model_name == 'Densenet': return build_densenet()
    if model_name == 'Densenet121': return build_densenet121()


@model_builder_ingredient.capture(prefix='densenet')
def build_densenet(_log, growth_rate, block_config, num_init_features, bn_size, drop_rate, memory_efficient, num_classes):
    densenet = DenseNet(growth_rate, block_config, num_init_features, bn_size, drop_rate, num_classes, memory_efficient)

    return nn.Sequential(
        densenet,
        nn.Sigmoid() if num_classes == 1 else nn.Softmax()
    )

@model_builder_ingredient.capture(prefix='densenet121')
def build_densenet121(_log, pretrained, frozen, num_classes):
    densenet = densenet121(pretrained, num_classes=1000 if pretrained else num_classes)

    if pretrained and num_classes != 1000:
        num_ftrs = densenet.classifier.in_features
        densenet.classifier = nn.Linear(num_ftrs, num_classes)

    if frozen:
        for param in densenet.parameters():
            param.requires_grad = False
        densenet.classifier.requires_grad = True

    return nn.Sequential(
        densenet,
        nn.Sigmoid() if num_classes == 1 else nn.Softmax()
    )