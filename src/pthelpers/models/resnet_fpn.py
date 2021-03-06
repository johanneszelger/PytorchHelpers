'''FPN in PyTorch.
See the paper "Feature Pyramid Networks for Object Detection" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torchvision.models import resnet


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                  padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                  padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4


    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
            )


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(FPN, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        self.avgPoolP2 = nn.AdaptiveAvgPool2d((1, 1))
        self.avgPoolP3 = nn.AdaptiveAvgPool2d((1, 1))
        self.avgPoolP4 = nn.AdaptiveAvgPool2d((1, 1))
        self.avgPoolP5 = nn.AdaptiveAvgPool2d((1, 1))

        # Classifiers
        self.classifierP2 = nn.Sequential(nn.Dropout(0.5), nn.Linear(256, num_classes), nn.ReLU())
        self.classifierP3 = nn.Sequential(nn.Dropout(0.5), nn.Linear(256, num_classes), nn.ReLU())
        self.classifierP4 = nn.Sequential(nn.Dropout(0.5), nn.Linear(256, num_classes), nn.ReLU())
        self.classifierP5 = nn.Sequential(nn.Dropout(0.5), nn.Linear(256, num_classes), nn.ReLU())

        self.classifier = nn.Linear(4, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y


    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        p5 = self.avgPoolP5(p5)
        p4 = self.avgPoolP5(p4)
        p3 = self.avgPoolP5(p3)
        p2 = self.avgPoolP5(p2)

        p5 = torch.flatten(p5, 1)
        p4 = torch.flatten(p4, 1)
        p3 = torch.flatten(p3, 1)
        p2 = torch.flatten(p2, 1)

        pred2 = self.classifierP2(p2)
        pred3 = self.classifierP2(p3)
        pred4 = self.classifierP2(p4)
        pred5 = self.classifierP2(p5)

        pred = self.classifier(torch.cat([pred2, pred3, pred4, pred5], dim=1))

        return pred


def resnet18_fpn(num_classes: int):
    """Constructs a ResNet-18 model. With a FPN
    """
    return FPN(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def resnet34_fpn(num_classes: int):
    """Constructs a ResNet-34 model. With a FPN
    """
    return FPN(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def resnet50_fpn(num_classes: int):
    """Constructs a ResNet-50 model. With a FPN
    """
    return FPN(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def resnet101_fpn(num_classes: int):
    """Constructs a ResNet-101 model. With a FPN
    """
    return FPN(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


def resnet152_fpn(num_classes: int):
    """Constructs a ResNet-152 model. With a FPN
    """
    return FPN(Bottleneck, [[3, 8, 36, 3]], num_classes=num_classes)


def test():
    net = resnet50_fpn()
    fms = net(Variable(torch.randn(1, 3, 600, 900)))
    for fm in fms:
        print(fm.size())
