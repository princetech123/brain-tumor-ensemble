import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    ResNet50_Weights, 
    EfficientNet_B3_Weights, 
    ViT_B_16_Weights
)
from typing import Dict, Any, Tuple

class ResNet50Backbone(nn.Module):
    """Wrapper for ResNet50 feature extraction and classification."""
    def __init__(self, pretrained: bool = True, num_classes: int = 1):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.model = models.resnet50(weights=weights)
        self.feature_dim = self.model.fc.in_features
        
        # Replace fully connected layer with Identity to extract features
        self.model.fc = nn.Identity()
        
        # Classifier for standalone backbone training
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def freeze_backbone(self, freeze: bool = True):
        """Freezes or unfreezes the entire ResNet50 feature extractor."""
        for param in self.model.parameters():
            param.requires_grad = not freeze
            
    def unfreeze_last_blocks(self):
        """Unfreezes only layer4 of ResNet50 for fine-tuning."""
        self.freeze_backbone(True)
        for param in self.model.layer4.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.model(x)  # Shape: (B, 2048)
        if return_features:
            return features
        logits = self.classifier(features)
        return logits, features


class EfficientNetB3Backbone(nn.Module):
    """Wrapper for EfficientNet-B3 feature extraction and classification."""
    def __init__(self, pretrained: bool = True, num_classes: int = 1):
        super().__init__()
        weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
        self.model = models.efficientnet_b3(weights=weights)
        self.feature_dim = self.model.classifier[1].in_features
        
        # Replace classifier head with Identity
        self.model.classifier = nn.Identity()
        
        # Classifier for standalone training
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def freeze_backbone(self, freeze: bool = True):
        """Freezes or unfreezes the entire EfficientNet backbone."""
        for param in self.model.parameters():
            param.requires_grad = not freeze
            
    def unfreeze_last_blocks(self):
        """Unfreezes the last block of EfficientNet features (features[7] and features[8]) for fine-tuning."""
        self.freeze_backbone(True)
        # EfficientNet-B3 features has 9 blocks (0 to 8)
        for block in [self.model.features[7], self.model.features[8]]:
            for param in block.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.model(x)  # Shape: (B, 1536)
        if return_features:
            return features
        logits = self.classifier(features)
        return logits, features


class ViTBackbone(nn.Module):
    """Wrapper for Vision Transformer ViT-B/16 feature extraction and classification."""
    def __init__(self, pretrained: bool = True, num_classes: int = 1):
        super().__init__()
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        self.model = models.vit_b_16(weights=weights)
        self.feature_dim = self.model.heads.head.in_features
        
        # Replace heads with Identity to output cls token features
        self.model.heads = nn.Identity()
        
        # Standalone classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def freeze_backbone(self, freeze: bool = True):
        """Freezes or unfreezes the entire ViT backbone."""
        for param in self.model.parameters():
            param.requires_grad = not freeze
            
    def unfreeze_last_blocks(self):
        """Unfreezes the final two encoder blocks in ViT for fine-tuning."""
        self.freeze_backbone(True)
        # ViT-B/16 has 12 encoder layers (0 to 11)
        for layer in [self.model.encoder.layers[10], self.model.encoder.layers[11]]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        # ViT-B/16 strictly requires 224x224 input (fixed patch size).
        # Auto-resize if the input has a different spatial dimension.
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        features = self.model(x)  # Shape: (B, 768)
        if return_features:
            return features
        logits = self.classifier(features)
        return logits, features


def load_backbone(model_name: str, pretrained: bool = True) -> nn.Module:
    """Factory function to load backbones by name."""
    name_lower = model_name.lower()
    if "resnet" in name_lower:
        return ResNet50Backbone(pretrained=pretrained)
    elif "efficientnet" in name_lower:
        return EfficientNetB3Backbone(pretrained=pretrained)
    elif "vit" in name_lower:
        return ViTBackbone(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
