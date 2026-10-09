import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
num_classes = len(class_names)

print("Classes:", class_names)

# =========================
# Load Best Single Model
# =========================
model = timm.create_model(
    "convnext_tiny",
    pretrained=False,
    num_classes=4
)

model.load_state_dict(torch.load("convnext_model.pth", map_location=device))
model = model.to(device)
model.eval()

print("ConvNeXt model loaded.")

# =========================
# Validation Evaluation
# =========================
all_preds = []
all_labels = []
all_probs = []

correct = 0
total = 0

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)

        _, predicted = torch.max(probs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

val_acc = 100 * correct / total

# =========================
# Show Best Train Accuracy
# (Enter manually from training logs)
# =========================
best_train_acc = "Not Recorded in train.py"

print("\n==============================")
print("BEST MODEL RESULTS")
print("==============================")
# print("Best Train Accuracy :", best_train_acc)
print(f"Best Validation Accuracy : {val_acc:.2f}%")
print("==============================")

# =========================
# Classification Report
# =========================
print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=class_names))

# =========================
# Confusion Matrix
# =========================
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Confusion Matrix - ConvNeXt")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================
# ROC Curve (One vs Rest)
# =========================
all_labels_bin = label_binarize(all_labels, classes=[0,1,2,3])
all_probs = np.array(all_probs)

plt.figure(figsize=(8,6))

for i in range(num_classes):
    fpr, tpr, _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")

plt.plot([0,1], [0,1], linestyle="--")
plt.title("ROC Curve - ConvNeXt")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=300, bbox_inches="tight")
plt.show()