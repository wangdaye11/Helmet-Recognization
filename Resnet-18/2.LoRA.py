import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from base_model.resnet18_base import get_resnet18_base
from lora.layers import LoRAConv2d
import os
import time
import matplotlib.pyplot as plt

# ----- Configurations -----
num_classes = 4
batch_size = 32
epochs = 5
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Define ResNet18 + LoRA -----
class ResNet18WithLoRA(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = get_resnet18_base(num_classes=num_classes, pretrained=True)
        for name, module in self.model.named_children():
            if name == "layer4":
                for block in module:
                    block.conv1 = LoRAConv2d(block.conv1, r=4, alpha=1.0)
                    block.conv2 = LoRAConv2d(block.conv2, r=4, alpha=1.0)

    def forward(self, x):
        return self.model(x)

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

# ----- Initialize model -----
model = ResNet18WithLoRA(num_classes=num_classes).to(device)

# --------- Model statistics ---------
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params} / {total_params}")

# ----- Optimizer and loss -----
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# ----- Metrics log -----
train_acc_history = []
test_acc_history = []
train_loss_history = []
test_loss_history = []

# ----- Training loop -----
start_time = time.time()

for epoch in range(epochs):
    model.train()
    train_correct = 0
    train_total = 0
    train_loss_sum = 0.0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)

    train_acc = 100 * train_correct / train_total
    train_loss = train_loss_sum / len(train_loader)
    train_acc_history.append(train_acc)
    train_loss_history.append(train_loss)

    # ----- Evaluation on test -----
    model.eval()
    test_correct = 0
    test_total = 0
    test_loss_sum = 0.0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            test_loss_sum += loss.item()
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)

    test_acc = 100 * test_correct / test_total
    test_loss = test_loss_sum / len(test_loader)
    test_acc_history.append(test_acc)
    test_loss_history.append(test_loss)

    print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

elapsed = time.time() - start_time
print(f"Total training time: {elapsed:.2f} seconds")

# ----- Save model -----
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/resnet18_lora.pth")

# ----- Plot: Loss -----
plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs + 1), train_loss_history, label="Train Loss", marker='o')
plt.plot(range(1, epochs + 1), test_loss_history, label="Test Loss", marker='s')
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(range(1, epochs + 1))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("checkpoints/lora_loss_curve.png")
plt.show()

# ----- Plot: accuracy -----
plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs + 1), train_acc_history, label="Train Accuracy", marker='o')
plt.plot(range(1, epochs + 1), test_acc_history, label="Test Accuracy", marker='s')
plt.title("Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.xticks(range(1, epochs + 1))
plt.ylim(50, 100)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("checkpoints/lora_accuracy_curve.png")
plt.show()
