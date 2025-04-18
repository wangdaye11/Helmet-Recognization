import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from base_model.resnet18_base import get_resnet18_base
import os
import time

# ----- 配置参数 -----
num_classes = 4
batch_size = 32
epochs = 5
lr = 1e-4  # 全参数微调建议稍微小一点
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- 加载模型（不冻结任何层） -----
model = get_resnet18_base(num_classes=num_classes, pretrained=True)
model = model.to(device)

# ----- 加载训练数据 -----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder("../dataset/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
class_names = train_dataset.classes
print("类别索引映射：", class_names)

# ----- 全部参数参与训练 -----
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

start_time = time.time()

# 🔍 打印参数数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ 全参数微调：{trainable_params:,} / {total_params:,} 可训练")

# ----- 训练函数 -----
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
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

# ----- 主训练循环 -----
for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"[Epoch {epoch+1}] ✅ Loss: {train_loss:.4f} | Accuracy: {train_acc:.2f}%")

end_time = time.time()
elapsed = end_time - start_time
print(f"⏱️ 总训练时间: {elapsed:.2f} 秒 ({elapsed/60:.2f} 分钟)")

# ----- 保存模型 -----
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/resnet18_full_finetune_4class.pth")
