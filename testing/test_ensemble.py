import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.swin import SwinClassifier
from models.convnext import ConvNeXtClassifier
from ensemble import fuse

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
swin = SwinClassifier().to(DEVICE)
conv = ConvNeXtClassifier().to(DEVICE)

swin.load_state_dict(torch.load("weights/best1_swin.pth", map_location=DEVICE))
conv.load_state_dict(torch.load("weights/best_convnext.pth", map_location=DEVICE))

swin.eval()
conv.eval()

# IMPORTANT: must match training preprocessing
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

test_data = datasets.ImageFolder("dataset/HAM10000/test1", transform=transform)
loader = DataLoader(test_data, batch_size=16, shuffle=False)

correct = 0
total = 0

with torch.no_grad():
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        out1 = swin(x)
        out2 = conv(x)

        final = fuse(out1, out2)
        preds = final.argmax(1)

        correct += (preds == y).sum().item()
        total += y.size(0)

acc = 100 * correct / total
print(f"\nEnsemble Accuracy: {acc:.2f}%")
