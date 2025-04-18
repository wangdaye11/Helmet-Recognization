from base_model.resnet18_base import get_resnet18_base
from lora.layers import LoRAConv2d
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim

# 定义集成 LoRA 的 ResNet18 模型
class ResNet18WithLoRA(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18WithLoRA, self).__init__()
        self.model = get_resnet18_base(num_classes=num_classes, pretrained=True)

        for name, module in self.model.named_children():
            if "layer4" in name:
                for block in module:
                    block.conv1 = LoRAConv2d(block.conv1, r=4, alpha=1.0)
                    block.conv2 = LoRAConv2d(block.conv2, r=4, alpha=1.0)

    def forward(self, x):
        return self.model(x)

# 数据增强与加载
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder("../dataset/train", transform=transform)
val_dataset = datasets.ImageFolder("../dataset/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class_names = train_dataset.classes

# 初始化模型
model = ResNet18WithLoRA(num_classes=len(class_names))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 优化器与损失函数
lora_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(lora_params, lr=1e-3)
criterion = nn.CrossEntropyLoss()

# 训练与验证函数
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

def evaluate(model, dataloader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, acc

# 训练主循环
epochs = 5
for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    print(f"[Epoch {epoch+1}] Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

# 保存训练好的 LoRA 模型参数
torch.save(
    {k: v for k, v in model.state_dict().items() if v.requires_grad},
    "checkpoints/resnet18_lora_only.pth"
)
