import torch.nn as nn
from torchvision import models


def get_model(num_classes=2, arch="efficientnet_b0"):
    if arch == "swin_v2_t":
        model = models.swin_v2_t(weights=models.Swin_V2_T_Weights.IMAGENET1K_V1)
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
        return model

    if arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    if arch == "efficientnet_b3":
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unknown model architecture: {arch}")
