# models/swin.py
import timm
import torch.nn as nn

class SwinClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.model = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)
