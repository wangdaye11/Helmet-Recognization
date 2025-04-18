import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from base_model.resnet18_base import get_resnet18_base
import os
import time

# ----- Config -----
num_classes = 4
batch_size = 32
epochs = 5
lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Load base model -----
model = get_resnet18_base(num_classes=num_classes, pretrained=True)

# Freeze all layers first
for param in model.parameters():
    param.requires_grad = False

# Only unfreeze layer4.conv1 and conv2
for name, module in model.named_modules():
    if ("layer4.0.conv1" in name or "layer4.0.conv2" in name or
        "layer4.1.conv1" in name or "layer4.1.conv2" in name):
        for param in module.parameters():
            param.requires_grad = True

model = model.to(device)

# ----- Load dataset -----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
train_dataset = datasets.ImageFolder("../dataset/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
class_names = train_dataset.classes
print("Class mapping:", class_names)

# ----- Optimizer, Loss -----
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(trainable_params, lr=lr)
criterion = nn.CrossEntropyLoss()

# 🔍 Stats
total_params = sum(p.numel() for p in model.parameters())
trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ Trainable parameters: {trainable_params_count:,} / {total_params:,}")
print(f"🖥️  Using device: {device}")

# ----- Train -----
start_time = time.time()

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

for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"[Epoch {epoch+1}] ✅ Loss: {train_loss:.4f} | Accuracy: {train_acc:.2f}%")

end_time = time.time()
elapsed = end_time - start_time
print(f"⏱️ Total training time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

# ----- Save -----
os.makedirs("checkpoints", exist_ok=True)
torch.save(
    {k: v for k, v in model.state_dict().items() if v.requires_grad},
    "checkpoints/resnet18_layer4_only_4class.pth"
)
