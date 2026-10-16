import os
import torch
from collections import Counter
from torchvision import datasets
from torchvision.utils import save_image
from train_gan import Generator, LATENT_DIM, DEVICE

TRAIN_ROOT = "../dataset/HAM10000/images"
TARGET_RATIO = 0.6   # balance to 60% of largest class

dataset = datasets.ImageFolder(TRAIN_ROOT)
counts = Counter(dataset.targets)
classes = dataset.classes

print("Original distribution:")
for i,c in enumerate(classes):
    print(c, counts[i])

max_count = max(counts.values())
target = int(max_count * TARGET_RATIO)

print("\nTarget per class:", target)

for idx, cname in enumerate(classes):

    current = counts[idx]

    if current >= target:
        print(f"Skipping {cname}")
        continue

    needed = target - current
    print(f"Generating {needed} images for {cname}")

    gen_path = f"generator_{cname}.pth"
    if not os.path.exists(gen_path):
        print(f"Train GAN first for {cname}")
        continue

    G = Generator().to(DEVICE)
    G.load_state_dict(torch.load(gen_path, map_location=DEVICE))
    G.eval()

    save_dir = os.path.join(TRAIN_ROOT, cname)

    for i in range(needed):
        z = torch.randn(1, LATENT_DIM).to(DEVICE)
        img = (G(z) + 1) / 2
        save_image(img, f"{save_dir}/gan_{cname}_{i}.jpg")

print("\nDataset Balanced")
