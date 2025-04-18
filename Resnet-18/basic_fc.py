import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from base_model.resnet18_base import get_resnet18_base  # 你写的模型
import os

# ----- 配置参数 -----
num_classes = 2
batch_size = 32
epochs = 5
lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- 加载模型 -----
model = get_resnet18_base(num_classes=num_classes, pretrained=True)

# 冻结除 fc 外的所有参数
for name, param in model.named_parameters():
    if not name.startswith("fc"):
        param.requires_grad = False

model = model.to(device)

# ----- 加载数据 -----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder("../dataset/train", transform=transform)
val_dataset = datasets.ImageFolder("../dataset/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

class_names = train_dataset.classes
print("类别索引映射：", class_names)

# ----- 损失函数与优化器 -----
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

# ----- 训练 -----
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"[Epoch {epoch+1}] Training Loss: {total_loss:.4f}")

    # ----- 验证 -----
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"[Epoch {epoch+1}] ✅ Validation Accuracy: {acc:.2f}%\n")

# ----- 保存模型 -----
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/resnet18_fc_only.pth")
