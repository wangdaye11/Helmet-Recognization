import torch.nn as nn
from torchvision.models import resnet18

def get_resnet18_base(num_classes=2, pretrained=True):
    model = resnet18(pretrained=pretrained)
    # 替换分类头：适配你自己的任务（如2分类）
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model