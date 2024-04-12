import torch.nn as nn
from .mlp import MLP
from .resnet import wrn28_10
from .linear import LinearNet
import torchvision.models as models


def build_model(params):
    if params.architecture == "mlp":
        return MLP(params)
    elif params.architecture == "linear":
        return LinearNet(params)
    elif params.architecture == "wrn28_10":
        return wrn28_10(10, dropout=False)
    elif params.architecture=="densenet121":
        densenet121 = models.densenet121(pretrained=False)
        num_ftrs = densenet121.classifier.in_features
        densenet121.classifier = nn.Linear(num_ftrs, 100)
        return densenet121
    elif params.architecture == "vgg16":
        return models.vgg16(pretrained=False, num_classes=params.num_classes)
    elif params.architecture == "smallnet":
        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(128, params.num_classes, bias=True),
        )