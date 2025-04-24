import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from base_model.resnet18_base import get_resnet18_base
import os
import time
from thop import profile, clever_format
import matplotlib.pyplot as plt

# Configuration
num_classes = 4
batch_size = 32
epochs = 5
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load base model
model = get_resnet18_base(num_classes=num_classes, pretrained=True)

# Freeze all layers except the final fully connected layer
for name, param in model.named_parameters():
    if not name.startswith("fc"):
        param.requires_grad = False

model = model.to(device)

# Count total and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params} / {total_params}")

# Estimate FLOPs and parameters using a dummy input
dummy_input = torch.randn(1, 3, 224, 224).to(device)
macs, params = profile(model, inputs=(dummy_input,), verbose=False)
macs, params = clever_format([macs, params], "%.2f")
print(f"FLOPs: {macs}, Parameters: {params}")

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load datasets (train and test only)
train_dataset = datasets.ImageFolder("../dataset/train", transform=transform)
test_dataset = datasets.ImageFolder("../dataset/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class_names = train_dataset.classes
print("Class mapping:", class_names)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

# Accuracy history for plotting
train_acc_history = []

# Training loop
start_time = time.time()

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    average_loss = running_loss / len(train_loader)
    training_accuracy = 100 * correct_predictions / total_samples
    train_acc_history.append(training_accuracy)

    print(f"Epoch {epoch + 1}/{epochs} - Loss: {average_loss:.4f} - Train Acc: {training_accuracy:.2f}%")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total training time: {elapsed_time:.2f} seconds")

# Save the trained model
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/resnet18_fc_only.pth")

# Final test evaluation
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)

test_accuracy = 100 * test_correct / test_total
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Plot training accuracy curve only
plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs + 1), train_acc_history, label="Train Accuracy", marker='o')
plt.title("Train Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.xticks(range(1, epochs + 1))
plt.ylim(50, 100)  # <-- 设置 y 轴范围为 50 到 100
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("checkpoints/basic_fc_accuracy_curve.png")
plt.show()

