import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# Base ResNet18 loader
from base_model.resnet18_base import get_resnet18_base

# LoRA wrapper model
from lora.layers import LoRAConv2d


# ----- Define LoRA-enhanced model -----
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


# ----- Config -----
val_data_path = "../dataset/val"
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Data transforms -----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ----- Load val dataset -----
val_dataset = datasets.ImageFolder(val_data_path, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
class_names = val_dataset.classes
print("Class mapping:", class_names)

# ----- Model checkpoint paths -----
checkpoint_paths = {
    "FC-only": "checkpoints/resnet18_fc_only.pth",
    "LoRA": "checkpoints/resnet18_lora.pth",
    "Layer4-only": "checkpoints/resnet18_layer4.pth",
    "Full Finetune": "checkpoints/resnet18_full_finetune.pth"
}


# ----- Evaluation function -----
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total


# ----- Loop over each checkpoint and evaluate -----
results = {}

for name, path in checkpoint_paths.items():
    print(f"Evaluating: {name}")

    # --- Use correct model structure ---
    if "lora" in name.lower():
        model = ResNet18WithLoRA(num_classes=len(class_names))
    else:
        model = get_resnet18_base(num_classes=len(class_names), pretrained=False)

    # --- Load model weights safely ---
    state = torch.load(path, map_location=device)

    # If wrapped in a dict (e.g., {"model": ..., "meta": ...}), extract 'model'
    if isinstance(state, dict) and "model" in state:
        state_dict = state["model"]
    elif isinstance(state, dict):
        # Remove non-tensor keys like total_ops or meta
        state_dict = {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}
    else:
        raise ValueError(f"Unexpected checkpoint format in {path}")

    # Load weights (strict=False to tolerate partial mismatch if needed)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)

    # --- Evaluate on validation set ---
    acc = evaluate(model, val_loader, device)
    print(f"{name} Val Accuracy: {acc:.2f}%")
    results[name] = acc

# ----- Plot comparison -----
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values(), color='skyblue')
plt.title("Validation Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.ylim(50, 100)
plt.xticks(rotation=15)
for i, (k, v) in enumerate(results.items()):
    plt.text(i, v + 0.5, f"{v:.2f}%", ha='center')
plt.tight_layout()
plt.savefig("checkpoints/model_val_accuracy_comparison.png")
plt.show()