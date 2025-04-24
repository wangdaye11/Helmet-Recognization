import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from base_model.resnet18_base import get_resnet18_base
import os
import time
import matplotlib.pyplot as plt

# ----- Configuration -----
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

# Unfreeze only layer4 conv1 and conv2
for name, module in model.named_modules():
    if ("layer4.0.conv1" in name or "layer4.0.conv2" in name or
        "layer4.1.conv1" in name or "layer4.1.conv2" in name):
        for param in module.parameters():
            param.requires_grad = True

model = model.to(device)

# ----- Data transforms -----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ----- Load datasets -----
train_dataset = datasets.ImageFolder("../dataset/train", transform=transform)
test_dataset = datasets.ImageFolder("../dataset/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class_names = train_dataset.classes
print("Class mapping:", class_names)

# ----- Optimizer and loss -----
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(trainable_params, lr=lr)
criterion = nn.CrossEntropyLoss()

# ----- Statistics -----
total_params = sum(p.numel() for p in model.parameters())
trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params_count:,} / {total_params:,}")
print(f"Using device: {device}")

# ----- Accuracy logs -----
train_acc_history = []

# ----- Train function -----
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

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

# ----- Train loop -----
start_time = time.time()

for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    train_acc_history.append(train_acc)

    print(f"[Epoch {epoch+1}] Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

elapsed = time.time() - start_time
print(f"Total training time: {elapsed:.2f} seconds")

# ----- Final test -----
test_acc = evaluate(model, test_loader, device)
print(f"Test Accuracy: {test_acc:.2f}%")

# ----- Save model -----
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/resnet18_layer4.pth")

# ----- Plot training accuracy -----
plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs + 1), train_acc_history, label="Train Accuracy", marker='o')
plt.title("Train Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.xticks(range(1, epochs + 1))
plt.ylim(50, 100)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("checkpoints/layer4_accuracy_curve.png")
plt.show()
