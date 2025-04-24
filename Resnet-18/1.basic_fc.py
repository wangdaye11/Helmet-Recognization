import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import time
import matplotlib.pyplot as plt
from base_model.resnet18_base import get_resnet18_base

# --------- Configuration ---------
num_classes = 4
batch_size = 32
epochs = 5
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------- Load model ---------
model = get_resnet18_base(num_classes=num_classes, pretrained=True)

# Freeze all layers except the fully connected head
for name, param in model.named_parameters():
    if not name.startswith("fc"):
        param.requires_grad = False

model = model.to(device)

# --------- Model statistics ---------
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params} / {total_params}")

# --------- Dataset loading ---------
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

# --------- Optimizer and loss ---------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

# --------- Logs ---------
train_acc_history = []
test_acc_history = []
train_loss_history = []
test_loss_history = []

# --------- Evaluation function ---------
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy

# --------- Training loop ---------
start_time = time.time()

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    test_loss, test_acc = evaluate(model, test_loader)

    # Save to history
    train_loss_history.append(train_loss)
    test_loss_history.append(test_loss)
    train_acc_history.append(train_acc)
    test_acc_history.append(test_acc)

    print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

elapsed_time = time.time() - start_time
print(f"Total training time: {elapsed_time:.2f} seconds")

# --------- Save trained model ---------
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/resnet18_fc_only.pth")

# --------- Plot: Loss ---------
plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs + 1), train_loss_history, label="Train Loss", marker='o')
plt.plot(range(1, epochs + 1), test_loss_history, label="Test Loss", marker='s')
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(range(1, epochs + 1))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("checkpoints/fc_loss_curve.png")
plt.show()

# --------- Plot: Accuracy ---------
plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs + 1), train_acc_history, label="Train Accuracy", marker='o')
plt.plot(range(1, epochs + 1), test_acc_history, label="Test Accuracy", marker='s')
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.xticks(range(1, epochs + 1))
plt.ylim(50, 100)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("checkpoints/fc_accuracy_curve.png")
plt.show()
