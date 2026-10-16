import torch
from models.swin import SwinClassifier
from models.convnext import ConvNeXtClassifier
from ensemble import fuse
from gradcam import GradCAMPlusPlus
from torchvision import transforms
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

swin = SwinClassifier().to(device)
conv = ConvNeXtClassifier().to(device)

swin.load_state_dict(torch.load("weights/best1_swin.pth", map_location=device))
conv.load_state_dict(torch.load("weights/best_convnext.pth", map_location=device))

swin.eval()
conv.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


cam = GradCAMPlusPlus(swin, swin.model.patch_embed.proj)

def predict(img_np):
    x = transform(Image.fromarray(img_np)).unsqueeze(0).to(device)

    out1 = swin(x)
    out2 = conv(x)

    final = fuse(out1, out2)   # <--- your ensemble.py is used HERE

    pred = final.argmax().item()
    heat = cam.generate(x, pred)
    overlay = cam.overlay(img_np, heat)

    return pred, overlay
