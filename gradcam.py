# gradcam.py
import torch
import cv2
import numpy as np

class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, x, class_idx):
        self.model.zero_grad()
        out = self.model(x)
        score = out[:, class_idx]
        score.backward()

        grads = self.gradients
        acts = self.activations

        weights = torch.mean(grads, dim=(2,3), keepdim=True)
        cam = torch.sum(weights * acts, dim=1)

        cam = torch.relu(cam)
        cam = cam[0].detach().cpu().numpy()
        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        return cam

    def overlay(self, img, cam):
        heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        overlay = 0.4 * heat + 0.6 * img
        return overlay.astype(np.uint8)
