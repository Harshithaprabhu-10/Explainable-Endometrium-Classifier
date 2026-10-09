
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import timm

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# =========================
# Transform
# =========================
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =========================
# Dataset
# =========================
val_data = datasets.ImageFolder("dataset/val", transform=val_tf)
val_loader = DataLoader(val_data, batch_size=8, shuffle=False)

class_names = val_data.classes
print("Classes:", class_names)

# Class Index:
# 0 = EA
# 1 = EH
# 2 = EP
# 3 = NE

# =========================
# Load ConvNeXt
# =========================
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import timm

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# =========================
# Transform
# =========================
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =========================
# Dataset
# =========================
val_data = datasets.ImageFolder("dataset/val", transform=val_tf)
val_loader = DataLoader(val_data, batch_size=8, shuffle=False)

print("Classes:", val_data.classes)

# =========================
# Load ConvNeXt
# =========================
model_conv = timm.create_model(
    "convnext_tiny",
    pretrained=False,
    num_classes=4
)

model_conv.load_state_dict(torch.load("convnext_model.pth", map_location=device))
model_conv = model_conv.to(device)
model_conv.eval()

print("ConvNeXt loaded.")

# =========================
# Load ResNet50
# =========================
model_res = models.resnet50()
model_res.fc = nn.Sequential(
    nn.Dropout(0.75),
    nn.Linear(model_res.fc.in_features, 4)
)

checkpoint = torch.load("resnet_model.pth", map_location=device)
model_res.load_state_dict(checkpoint["model"])

model_res = model_res.to(device)
model_res.eval()

print("ResNet loaded.")

# =========================
# Store all outputs once
# =========================
all_conv = []
all_res = []
all_labels = []

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)

        out_conv = torch.softmax(model_conv(images), dim=1).cpu()
        out_res  = torch.softmax(model_res(images), dim=1).cpu()

        all_conv.append(out_conv)
        all_res.append(out_res)
        all_labels.append(labels)

all_conv = torch.cat(all_conv, dim=0)
all_res = torch.cat(all_res, dim=0)
all_labels = torch.cat(all_labels, dim=0)

# =========================
# Grid Search
# =========================
best_acc = 0
best_w = 0

print("\nSearching best weights...\n")

for i in range(5, 100, 5):

    w_conv = i / 100
    w_res = 1 - w_conv

    final_out = (w_conv * all_conv) + (w_res * all_res)

    preds = torch.argmax(final_out, dim=1)

    correct = (preds == all_labels).sum().item()
    total = len(all_labels)

    acc = 100 * correct / total

    print(f"ConvNeXt={w_conv:.2f} | ResNet={w_res:.2f} --> {acc:.2f}%")

    if acc > best_acc:
        best_acc = acc
        best_w = w_conv

# =========================
# Final Result
# =========================
print("\n======================")
print(f"Best Weight Found:")
print(f"ConvNeXt = {best_w:.2f}")
print(f"ResNet   = {1-best_w:.2f}")
print(f"Best Accuracy = {best_acc:.2f}%")
print("======================")
