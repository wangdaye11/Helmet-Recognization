import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from sklearn.metrics import f1_score

from base_model.resnet18_base import get_resnet18_base
from lora.layers import LoRAConv2d

# Define ResNet18 with LoRA inserted into layer4
class ResNet18WithLoRA(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = get_resnet18_base(num_classes=num_classes, pretrained=False)
        for name, module in self.model.named_children():
            if name == "layer4":
                for block in module:
                    block.conv1 = LoRAConv2d(block.conv1, r=4, alpha=1.0)
                    block.conv2 = LoRAConv2d(block.conv2, r=4, alpha=1.0)

    def forward(self, x):
        return self.model(x)

# ----------- Configuration -----------
val_data_path = "../dataset/val"
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transforms (same as used during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load validation dataset
val_dataset = datasets.ImageFolder(val_data_path, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
class_names = val_dataset.classes
print("Class mapping:", class_names)

# Paths to model checkpoints
checkpoint_paths = {
    "FC-only": "checkpoints/resnet18_fc_only.pth",
    "LoRA": "checkpoints/resnet18_lora.pth",
    "Layer4-only": "checkpoints/resnet18_layer4.pth",
    "Full Finetune": "checkpoints/resnet18_full_finetune.pth"
}

# ----------- Evaluation Function -----------
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    return accuracy, f1_macro, f1_weighted

# ----------- Result Dictionaries -----------
acc_results = {}
f1_macro_results = {}
f1_weighted_results = {}

# ----------- Load and Evaluate Each Model -----------
for name, path in checkpoint_paths.items():
    print(f"\nEvaluating: {name}")

    # Use LoRA model if specified, otherwise use base ResNet
    if "lora" in name.lower():
        model = ResNet18WithLoRA(num_classes=len(class_names))
    else:
        model = get_resnet18_base(num_classes=len(class_names), pretrained=False)

    # Load model weights
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state_dict = state["model"]
    elif isinstance(state, dict):
        state_dict = {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}
    else:
        raise ValueError(f"Checkpoint format not supported: {path}")

    # Load weights (strict=False allows partial loading)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)

    # Run evaluation
    acc, f1_macro, f1_weighted = evaluate(model, val_loader, device)
    print(f"{name} - Accuracy: {acc:.2f}% | F1 (macro): {f1_macro:.4f} | F1 (weighted): {f1_weighted:.4f}")

    acc_results[name] = acc
    f1_macro_results[name] = f1_macro
    f1_weighted_results[name] = f1_weighted

# ----------- Plot Accuracy Bar Chart -----------
plt.figure(figsize=(10, 6))
plt.bar(acc_results.keys(), acc_results.values(), color='skyblue')
plt.title("Validation Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.ylim(50, 100)
plt.xticks(rotation=15)
for i, (k, v) in enumerate(acc_results.items()):
    plt.text(i, v + 0.5, f"{v:.2f}%", ha='center')
plt.tight_layout()
plt.savefig("checkpoints/val_accuracy_comparison.png")
plt.show()

# ----------- Plot Macro F1 Score Bar Chart -----------
plt.figure(figsize=(10, 6))
plt.bar(f1_macro_results.keys(), f1_macro_results.values(), color='orange')
plt.title("Validation Macro F1 Score Comparison")
plt.ylabel("F1 Score")
plt.ylim(0.0, 1.0)
plt.xticks(rotation=15)
for i, (k, v) in enumerate(f1_macro_results.items()):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
plt.tight_layout()
plt.savefig("checkpoints/val_f1_macro_comparison.png")
plt.show()
