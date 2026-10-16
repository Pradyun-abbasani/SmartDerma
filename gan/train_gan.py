import os
import sys
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# -------- Detect if script is run directly --------
if len(sys.argv) > 1:
    CLASS_NAME = sys.argv[1]   # used when running manually
else:
    CLASS_NAME = None          # used when imported by auto_balance

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LATENT_DIM = 100
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 150

# -------- Transform --------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# -------- Custom Dataset --------
class SingleClassDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

# -------- Generator --------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 3*IMG_SIZE*IMG_SIZE),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.net(z)
        return x.view(-1, 3, IMG_SIZE, IMG_SIZE)

# -------- Discriminator --------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*IMG_SIZE*IMG_SIZE, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ==========================================================
# Only run training if script is executed directly
# ==========================================================
if CLASS_NAME is not None:

    DATA_PATH = f"../dataset/HAM10000/images/{CLASS_NAME}"

    dataset = SingleClassDataset(DATA_PATH, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Loaded {len(dataset)} images from {CLASS_NAME}")

    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)

    optG = torch.optim.Adam(G.parameters(), lr=0.0002)
    optD = torch.optim.Adam(D.parameters(), lr=0.0002)

    loss_fn = nn.BCELoss()

    for epoch in range(EPOCHS):
        for imgs in loader:

            imgs = imgs.to(DEVICE)
            bs = imgs.size(0)

            real = torch.ones(bs,1).to(DEVICE)
            fake = torch.zeros(bs,1).to(DEVICE)

            # Train Discriminator
            z = torch.randn(bs, LATENT_DIM).to(DEVICE)
            gen_imgs = G(z)

            lossD = loss_fn(D(imgs), real) + loss_fn(D(gen_imgs.detach()), fake)

            optD.zero_grad()
            lossD.backward()
            optD.step()

            # Train Generator
            lossG = loss_fn(D(gen_imgs), real)

            optG.zero_grad()
            lossG.backward()
            optG.step()

        print(f"{CLASS_NAME} Epoch {epoch+1}/{EPOCHS} | D {lossD.item():.3f} | G {lossG.item():.3f}")

    torch.save(G.state_dict(), f"generator_{CLASS_NAME}.pth")
    print(f"Saved generator_{CLASS_NAME}.pth")
