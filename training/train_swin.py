# training/train_swin.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.swin import SwinClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# -------- Correct Transform for Swin --------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# -------- Load GAN-balanced dataset --------
train_data = datasets.ImageFolder("dataset/HAM10000/images", transform=transform)
test_data  = datasets.ImageFolder("dataset/HAM10000/test", transform=transform)

print("Classes:", train_data.classes)
print("Training samples:", len(train_data))

train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=0)
test_loader  = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=0)

# -------- Model --------
model = SwinClassifier(num_classes=7).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

EPOCHS = 25   # increased for balanced dataset
best_acc = 0

for epoch in range(EPOCHS):

    # ---- TRAIN ----
    model.train()
    correct = 0
    total = 0
    running_loss = 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        out = model(x)
        loss = loss_fn(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    avg_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total

    # ---- TEST ----
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    test_acc = 100 * correct / total

    print(f"Epoch {epoch+1:02d} | Loss {avg_loss:.4f} | "
          f"Train Acc {train_acc:.2f}% | Test Acc {test_acc:.2f}%")

    # ---- SAVE BEST MODEL ----
    if test_acc > best_acc:
        best_acc = test_acc
        os.makedirs("weights", exist_ok=True)
        torch.save(model.state_dict(), "weights/best1_swin.pth")
        print("✅ Best model saved")

print("\nTraining Completed")
print("Best Test Accuracy:", best_acc)
