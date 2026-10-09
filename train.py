import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# =========================
# Settings
# =========================
epochs = 30
batch_size = 8
learning_rate = 0.00003

# =========================
# High Resolution Transforms
# =========================
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(256, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor()
])

val_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# =========================
# Dataset
# =========================
train_data = datasets.ImageFolder("dataset/train", transform=train_tf)
val_data = datasets.ImageFolder("dataset/val", transform=val_tf)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

print("Classes:", train_data.classes)

# =========================
# Model
# =========================
model = timm.create_model(
    "convnext_tiny",
    pretrained=True,
    num_classes=4
)

model = model.to(device)

# =========================
# Loss / Optimizer
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# =========================
# Resume Training
# =========================
start_epoch = 0
best_acc = 0

if os.path.exists("checkpoint.pth"):
    checkpoint = torch.load("checkpoint.pth", map_location=device)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    start_epoch = checkpoint["epoch"] + 1
    best_acc = checkpoint["best_acc"]

    print(f"Resuming from epoch {start_epoch}")

# =========================
# Training Loop
# =========================
for epoch in range(start_epoch, epochs):

    model.train()
    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # =====================
    # Validation
    # =====================
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total

    print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss:.2f} | Val Accuracy: {acc:.2f}%")

    # =====================
    # Save Best Model
    # =====================
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_model.pth")
        print("Best model saved.")

    # =====================
    # Save Checkpoint
    # =====================
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_acc": best_acc
    }, "checkpoint.pth")

# =========================
# Finished
# =========================
print("Training Finished")
print("Best Validation Accuracy:", best_acc)