import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from base_model.resnet18_base import get_resnet18_base
from lora.layers import LoRAConv2d
import os
import time

# ----- 配置参数 -----
num_classes = 4
batch_size = 32
epochs = 5
lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- 定义 LoRA 模型 -----
class ResNet18WithLoRA(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = get_resnet18_base(num_classes=num_classes, pretrained=True)

        for name, module in self.model.named_children():
            if "layer4" in name:
                for block in module:
                    block.conv1 = LoRAConv2d(block.conv1, r=4, alpha=1.0)
                    block.conv2 = LoRAConv2d(block.conv2, r=4, alpha=1.0)

    def forward(self, x):
        return self.model(x)

# ----- 加载数据 -----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder("../dataset/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
class_names = train_dataset.classes
print("类别索引映射：", class_names)

# ----- 初始化模型 -----
model = ResNet18WithLoRA(num_classes=len(class_names)).to(device)

start_time = time.time()

# 只训练 LoRA 参数
lora_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(lora_params, lr=lr)
criterion = nn.CrossEntropyLoss()

# 🔍 打印参数数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ 可训练参数数量: {trainable_params:,} / {total_params:,}")

# ----- 单轮训练函数 -----
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, acc

# ----- 训练主循环 -----
for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"[Epoch {epoch+1}] ✅ Loss: {train_loss:.4f} | Accuracy: {train_acc:.2f}%")

end_time = time.time()
elapsed = end_time - start_time
print(f"⏱️ 总训练时间: {elapsed:.2f} 秒 ({elapsed/60:.2f} 分钟)")

# ----- 保存 LoRA 微调后的模型权重 -----
os.makedirs("checkpoints", exist_ok=True)
torch.save(
    {k: v for k, v in model.state_dict().items() if v.requires_grad},
    "checkpoints/resnet18_lora_only_4class.pth"
)
