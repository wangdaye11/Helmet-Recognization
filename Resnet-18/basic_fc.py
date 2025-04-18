import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from base_model.resnet18_base import get_resnet18_base
import os
import time
from thop import profile, clever_format  # 🧠 计算模型FLOPs & Params

# ----- 配置参数 -----
num_classes = 4
batch_size = 32
epochs = 5
lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- 加载模型 -----
model = get_resnet18_base(num_classes=num_classes, pretrained=True)

# 冻结除 fc 外所有参数
for name, param in model.named_parameters():
    if not name.startswith("fc"):
        param.requires_grad = False

model = model.to(device)

# ✅ 可训练参数数量统计
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ 只训练 FC 层：{trainable_params:,} / {total_params:,} 可训练")

# 🧠 模型计算量（FLOPs）统计（假设输入是 1 x 3 x 224 x 224）
dummy_input = torch.randn(1, 3, 224, 224).to(device)
macs, params = profile(model, inputs=(dummy_input,), verbose=False)
macs, params = clever_format([macs, params], "%.2f")
print(f"📊 模型计算量（FLOPs）: {macs}, 参数量: {params}")

# ----- 数据预处理 -----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder("../dataset/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
class_names = train_dataset.classes
print("类别索引映射：", class_names)

# ----- 损失函数与优化器 -----
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

# ----- 训练 -----
start_time = time.time()  # ⏱️ 开始计时

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for imgs, labels in train_loader:
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

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

# ⏱️ 显示训练总耗时
end_time = time.time()
elapsed = end_time - start_time
print(f"⏱️ 总训练时间: {elapsed:.2f} 秒 ({elapsed/60:.2f} 分钟)")

# ----- 保存模型 -----
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/resnet18_fc_only_4class.pth")
