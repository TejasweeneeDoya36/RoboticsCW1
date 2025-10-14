# model.py
import torch
import torch.nn as nn
from torchvision import models

def get_mobilenet_v2(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Returns a MobileNetV2 model with the classifier head adapted to num_classes.
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
    in_feats = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_feats, num_classes)
    return model

def load_model(weights_path: str, num_classes: int, device: str = "cpu") -> nn.Module:
    model = get_mobilenet_v2(num_classes=num_classes, pretrained=False)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model
