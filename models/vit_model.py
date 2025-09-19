import os
import torch
import torch.nn as nn
from torchvision.models import vision_transformer

class ViTModel(nn.Module):
    def __init__(self, num_classes=39, pretrained=True):
        super(ViTModel, self).__init__()
        # Load pre-trained ViT-B_16 model
        self.model = vision_transformer.vit_b_16(weights='IMAGENET1K_V1' if pretrained else None)
        # Replace the head for your 39 classes
        self.model.heads.head = nn.Linear(self.model.heads.head.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def load_checkpoint(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, weights_only=True)
            try:
                self.model.load_state_dict(state_dict)
                print(f"Loaded checkpoint from {checkpoint_path}")
            except RuntimeError as e:
                print(f"Failed to load checkpoint due to mismatch: {e}. Initializing from scratch.")