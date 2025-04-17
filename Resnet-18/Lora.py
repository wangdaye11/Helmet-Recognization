from base_model.resnet18_base import get_resnet18_base
from lora.layers import LoRAConv2d

def get_resnet18_lora(num_classes=2):
    model = get_resnet18_base(num_classes=num_classes, pretrained=True)

    # 替换部分卷积层为 LoRAConv2d（示例只替换 layer4）
    for name, module in model.named_children():
        if "layer4" in name:
            for block in module:
                block.conv1 = LoRAConv2d(block.conv1, r=4, alpha=1.0)
                block.conv2 = LoRAConv2d(block.conv2, r=4, alpha=1.0)

    return model
