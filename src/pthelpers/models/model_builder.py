import wandb
from torch import nn
from torchvision.models import DenseNet, densenet121, DenseNet121_Weights, efficientnet_b0, EfficientNet_B0_Weights, resnet50, \
    efficientnet_b3, resnet152, ResNet50_Weights, efficientnet_b1, efficientnet_b2, resnet18, \
    resnet34, resnet101


class ModelBuilder():
    def __init__(self):
        if "model" not in wandb.config or "name" not in wandb.config["model"]:
            raise ValueError("To use the model builder, make sure to define a model dict in wandb config and define all necessary params "
                             "there, in particular model_name")


    def __freeze_model(self, model: nn.Module, classifier_name: str):
        for param in model.parameters():
            param.requires_grad = False
        # only unlock classifier
        for param in getattr(model, classifier_name).parameters():
            param.requires_grad = True


    def __add_sigmoid_or_softmax(self, model: nn.Module, num_classes: int):
        sq = nn.Sequential(
                model,
                nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=-1)
        )
        return sq


    def build_model(self):
        '''
        Builds a model, make sure to define a model dict in wandb config containing at least model_name and the models params.

        Valid names:
        DenseNet
        DenseNet121
        EfficientNetB0
        EfficientNetB1
        EfficientNetB2
        EfficientNetB3
        ResNet18
        ResNet34
        ResNet50
        ResNet101
        ResNet152

        :return: pytorch model
        '''
        model_name = wandb.config["model"]["name"]
        if model_name == 'DenseNet': return self.build_densenet()
        if model_name == 'DenseNet121': return self.build_densenet121()
        if model_name == 'EfficientNetB0': return self.build_efficientnetB0()
        if model_name == 'EfficientNetB1': return self.build_efficientnetB1()
        if model_name == 'EfficientNetB2': return self.build_efficientnetB2()
        if model_name == 'EfficientNetB3': return self.build_efficientnetB3()
        if model_name == 'ResNet18': return self.build_resnet18()
        if model_name == 'ResNet34': return self.build_resnet34()
        if model_name == 'ResNet50': return self.build_resnet50()
        if model_name == 'ResNet101': return self.build_resnet101()
        if model_name == 'ResNet152': return self.build_resnet152()


    def build_densenet(self):
        model_params = wandb.config["model"]

        growth_rate = model_params["growth_rate"]
        block_config = model_params["block_config"]
        num_init_features = model_params["num_init_features"]
        bn_size = model_params["bn_size"]
        drop_rate = model_params["drop_rate"]
        memory_efficient = model_params["memory_efficient"]
        num_classes = model_params["num_classes"]

        densenet = DenseNet(growth_rate, block_config, num_init_features, bn_size, drop_rate, num_classes, memory_efficient)

        return self.__add_sigmoid_or_softmax(densenet, num_classes)


    def build_densenet121(self):
        model_params = wandb.config["model"]

        weights = model_params["weights"] if "weights" in model_params else None
        drop_rate = model_params["drop_rate"] if "drop_rate" in model_params else 0
        num_classes = model_params["num_classes"] if "num_classes" in model_params else 1000
        frozen = model_params["frozen"] if "frozen" in model_params else False

        densenet = densenet121(weights=DenseNet121_Weights[weights] if weights else None,
                               num_classes=1000 if weights else num_classes, drop_rate=drop_rate)

        if weights and num_classes != 1000:
            num_ftrs = densenet.classifier.in_features
            densenet.classifier = nn.Linear(num_ftrs, num_classes)

        if frozen:
            self.__freeze_model(densenet, "classifier")

        return self.__add_sigmoid_or_softmax(densenet, num_classes)


    def __rebuild_efficientnet_classifier(self, efficientnet, num_classes, drop_rate):
        num_ftrs = efficientnet.classifier[1].in_features
        efficientnet.classifier = nn.Sequential(
                nn.Dropout(p=drop_rate, inplace=True),
                nn.Linear(num_ftrs, num_classes),
        )


    def __construct_standard_efficientnet(self, constructor):
        model_params = wandb.config["model"]

        weights = model_params["weights"] if "weights" in model_params else None
        drop_rate = model_params["drop_rate"] if "drop_rate" in model_params else 0.2
        num_classes = model_params["num_classes"] if "num_classes" in model_params else 1000
        frozen = model_params["frozen"] if "frozen" in model_params else False

        efficientnet = constructor(weights=EfficientNet_B0_Weights[weights] if weights else None,
                                   num_classes=1000 if weights else num_classes)

        if weights and num_classes != 1000:
            self.__rebuild_efficientnet_classifier(efficientnet, num_classes, drop_rate)

        if frozen:
            self.__freeze_model(efficientnet, "classifier")

        return self.__add_sigmoid_or_softmax(efficientnet, num_classes)


    def build_efficientnetB0(self):
        return self.__construct_standard_efficientnet(efficientnet_b0)


    def build_efficientnetB1(self):
        return self.__construct_standard_efficientnet(efficientnet_b1)


    def build_efficientnetB2(self):
        return self.__construct_standard_efficientnet(efficientnet_b2)


    def build_efficientnetB3(self):
        return self.__construct_standard_efficientnet(efficientnet_b3)


    def __rebuild_resnet_classifier(self, num_classes, resnet):
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Sequential(
                nn.Linear(num_ftrs, num_classes),
        )


    def __construct_standard_resnet(self, constructor):
        model_params = wandb.config["model"]
        weights = model_params["weights"] if "weights" in model_params else None
        num_classes = model_params["num_classes"] if "num_classes" in model_params else 1000
        frozen = model_params["frozen"] if "frozen" in model_params else False

        resnet = constructor(weights=ResNet50_Weights[weights] if weights else None,
                          num_classes=1000 if weights else num_classes)

        if weights and num_classes != 1000:
            self.__rebuild_resnet_classifier(num_classes, resnet)

        if frozen:
            self.__freeze_model(resnet, "fc")

        return self.__add_sigmoid_or_softmax(resnet, num_classes)


    def build_resnet18(self):
        return self.__construct_standard_resnet(resnet18)


    def build_resnet34(self):
        return self.__construct_standard_resnet(resnet34)


    def build_resnet50(self):
        return self.__construct_standard_resnet(resnet50)


    def build_resnet101(self):
        return self.__construct_standard_resnet(resnet101)


    def build_resnet152(self):
        return self.__construct_standard_resnet(resnet152)
