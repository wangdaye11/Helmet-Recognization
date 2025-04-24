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
from thop import profile, clever_format

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

# ----- Parameter stats -----
dummy_input = torch.randn(1, 3, 224, 224).to(device)
macs, params = profile(model, inputs=(dummy_input,), verbose=False)
macs, params = clever_format([macs, params], "%.2f")
print(f"FLOPs: {macs}, Parameters: {params}")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

# ----- Training setup -----
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

train_acc_history = []

# ----- Training loop -----
start_time = time.time()

for epoch in range(epochs):
    model.train()
    train_correct = 0
    total_samples = 0
    running_loss = 0.0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    train_acc = 100 * train_correct / total_samples
    avg_loss = running_loss / len(train_loader)

    train_acc_history.append(train_acc)
    print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f} - Train Acc: {train_acc:.2f}%")

elapsed = time.time() - start_time
print(f"Total training time: {elapsed:.2f} seconds")

# ----- Save LoRA-only weights -----
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/resnet18_lora.pth")


# ----- Final test evaluation -----
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, predicted = torch.max(outputs, 1)
        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)

test_acc = 100 * test_correct / test_total
print(f"Test Accuracy: {test_acc:.2f}%")

# ----- Plot training accuracy curve -----
plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs + 1), train_acc_history, label="Train Accuracy", marker='o')
plt.title("LoRA Training Accuracy (Train Only)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.xticks(range(1, epochs + 1))
plt.ylim(50, 100)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("checkpoints/lora_accuracy_curve.png")
plt.show()
