import torch.nn as nn
from torchvision.models import resnet18

def get_resnet18_base(num_classes=4, pretrained=True):
    model = resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
