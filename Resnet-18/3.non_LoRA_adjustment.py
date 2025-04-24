import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import time
import matplotlib.pyplot as plt
from base_model.resnet18_base import get_resnet18_base

# ----- Config -----
num_classes = 4
batch_size = 32
epochs = 5
lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Load base model -----
model = get_resnet18_base(num_classes=num_classes, pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Only unfreeze layer4 conv1/conv2
for name, module in model.named_modules():
    if "layer4.0.conv1" in name or "layer4.0.conv2" in name or \
       "layer4.1.conv1" in name or "layer4.1.conv2" in name:
        for param in module.parameters():
            param.requires_grad = True

model = model.to(device)

# ----- Dataset setup -----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
train_dataset = datasets.ImageFolder("../dataset/train", transform=transform)
test_dataset = datasets.ImageFolder("../dataset/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class_names = train_dataset.classes
print("Class mapping:", class_names)

# ----- Optimizer & Loss -----
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
criterion = nn.CrossEntropyLoss()

# ----- Stats -----
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

# ----- Logs -----
train_loss_history = []
test_loss_history = []
train_acc_history = []
test_acc_history = []

# ----- Training loop -----
start_time = time.time()

for epoch in range(epochs):
    model.train()
    train_loss_sum, train_correct, train_total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item()
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_loss = train_loss_sum / len(train_loader)
    train_acc = 100 * train_correct / train_total

    # ---- Test ----
    model.eval()
    test_loss_sum, test_correct, test_total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss_sum += loss.item()
            _, preds = torch.max(outputs, 1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)

    test_loss = test_loss_sum / len(test_loader)
    test_acc = 100 * test_correct / test_total

    train_loss_history.append(train_loss)
    test_loss_history.append(test_loss)
    train_acc_history.append(train_acc)
    test_acc_history.append(test_acc)

    print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
          f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

elapsed = time.time() - start_time
print(f"Total training time: {elapsed:.2f} seconds")

# ----- Save model -----
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/resnet18_layer4.pth")

# ----- Plot: loss -----
plt.figure()
plt.plot(range(1, epochs + 1), train_loss_history, label="Train Loss", marker='o')
plt.plot(range(1, epochs + 1), test_loss_history, label="Test Loss", marker='s')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Test Loss")
plt.legend()
plt.grid(True)
plt.savefig("checkpoints/layer4_loss_curve.png")
plt.show()

# ----- Plot: accuracy -----
plt.figure()
plt.plot(range(1, epochs + 1), train_acc_history, label="Train Accuracy", marker='o')
plt.plot(range(1, epochs + 1), test_acc_history, label="Test Accuracy", marker='s')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Train vs Test Accuracy")
plt.ylim(50, 100)
plt.legend()
plt.grid(True)
plt.savefig("checkpoints/layer4_accuracy_curve.png")
plt.show()
