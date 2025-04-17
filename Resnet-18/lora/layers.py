import torch.nn as nn

class LoRAConv2d(nn.Module):
    def __init__(self, original_conv, r=4, alpha=1.0):
        super().__init__()
        self.original = original_conv
        self.r = r
        self.alpha = alpha

        in_channels = original_conv.in_channels
        out_channels = original_conv.out_channels

        # 冻结原始卷积
        for param in self.original.parameters():
            param.requires_grad = False

        self.lora_A = nn.Conv2d(in_channels, r, kernel_size=1, bias=False)
        self.lora_B = nn.Conv2d(r, out_channels, kernel_size=1, bias=False)
        self.scaling = alpha / r

    def forward(self, x):
        return self.original(x) + self.scaling * self.lora_B(self.lora_A(x))
