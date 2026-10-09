# import torch
# import torch.nn as nn
# from torchvision import datasets, transforms, models
# from torch.utils.data import DataLoader
# import timm

# from sklearn.metrics import confusion_matrix, classification_report
# import seaborn as sns
# import matplotlib.pyplot as plt

# # =========================
# # Device
# # =========================
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using:", device)

# # =========================
# # Transform
# # =========================
# val_tf = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])

# # =========================
# # Dataset
# # =========================
# val_data = datasets.ImageFolder("dataset/val", transform=val_tf)
# val_loader = DataLoader(val_data, batch_size=8, shuffle=False)

# class_names = val_data.classes
# print("Classes:", class_names)

# # =========================
# # Load ConvNeXt
# # =========================
# model_conv = timm.create_model(
#     "convnext_tiny",
#     pretrained=False,
#     num_classes=4
# )

# model_conv.load_state_dict(torch.load("convnext_model.pth", map_location=device))
# model_conv = model_conv.to(device)
# model_conv.eval()

# print("ConvNeXt loaded.")

# # =========================
# # Load ResNet50
# # =========================
# model_res = models.resnet50()
# model_res.fc = nn.Sequential(
#     nn.Dropout(0.75),
#     nn.Linear(model_res.fc.in_features, 4)
# )

# checkpoint = torch.load("resnet_model.pth", map_location=device)
# model_res.load_state_dict(checkpoint["model"])

# model_res = model_res.to(device)
# model_res.eval()

# print("ResNet loaded.")

# # =========================
# # Predictions
# # =========================
# all_preds = []
# all_labels = []

# with torch.no_grad():

#     for images, labels in val_loader:

#         images = images.to(device)
#         labels = labels.to(device)

#         out_conv = torch.softmax(model_conv(images), dim=1)
#         out_res  = torch.softmax(model_res(images), dim=1)

#         # Best current ensemble
#         final_out = (0.75 * out_conv) + (0.25 * out_res)

#         _, predicted = torch.max(final_out, 1)

#         all_preds.extend(predicted.cpu().numpy())
#         all_labels.extend(labels.cpu().numpy())

# # =========================
# # Accuracy
# # =========================
# correct = sum([p == t for p, t in zip(all_preds, all_labels)])
# acc = 100 * correct / len(all_labels)

# print("\n======================")
# print(f"Ensemble Val Accuracy: {acc:.2f}%")
# print("======================")

# # =========================
# # Classification Report
# # =========================
# print("\nClassification Report:\n")
# print(classification_report(all_labels, all_preds, target_names=class_names))

# # =========================
# # Confusion Matrix
# # =========================
# cm = confusion_matrix(all_labels, all_preds)

# plt.figure(figsize=(8,6))
# sns.heatmap(
#     cm,
#     annot=True,
#     fmt="d",
#     cmap="Blues",
#     xticklabels=class_names,
#     yticklabels=class_names
# )

# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("ConvNeXt + ResNet Ensemble Confusion Matrix")
# plt.tight_layout()
# plt.show()

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
# Smart Ensemble Evaluation
# =========================
correct = 0
total = 0

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Model probabilities
        out_conv = torch.softmax(model_conv(images), dim=1)
        out_res  = torch.softmax(model_res(images), dim=1)

        # Base weighted ensemble
        final_out = (0.80 * out_conv) + (0.20 * out_res)

        # -------------------------
        # Smart Fixes
        # -------------------------
        for i in range(final_out.size(0)):

            probs = final_out[i]

            ep_prob = probs[2]
            ne_prob = probs[3]
            eh_prob = probs[1]

            # Rule 1:
            # If EP predicted but NE is very close -> choose NE
            if torch.argmax(probs).item() == 2:
                if abs(ep_prob - ne_prob) < 0.08:
                    probs[3] += 0.06

            # Rule 2:
            # Help EH when EH close to EA or EP
            if eh_prob > 0.28:
                probs[1] += 0.03

            final_out[i] = probs

        _, predicted = torch.max(final_out, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

acc = 100 * correct / total

print("\n======================")
print(f"Smart Ensemble Val Accuracy: {acc:.2f}%")
print("======================")