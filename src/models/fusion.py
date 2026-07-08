import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

class EnsembleFusionModel(nn.Module):
    """Ensemble model that fuses feature vectors from ResNet50, EfficientNet-B3, and ViT-B/16

    using a Multihead Self-Attention mechanism.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Feature dimensions from configuration
        self.resnet_dim = config["models"]["resnet"]["feature_dim"]
        self.effnet_dim = config["models"]["efficientnet"]["feature_dim"]
        self.vit_dim = config["models"]["vit"]["feature_dim"]
        
        # Shared attention projection dimension
        self.proj_dim = 512
        
        # Projection layers to map all features to a common dimension (512)
        self.resnet_proj = nn.Sequential(
            nn.Linear(self.resnet_dim, self.proj_dim),
            nn.LayerNorm(self.proj_dim),
            nn.ReLU()
        )
        self.effnet_proj = nn.Sequential(
            nn.Linear(self.effnet_dim, self.proj_dim),
            nn.LayerNorm(self.proj_dim),
            nn.ReLU()
        )
        self.vit_proj = nn.Sequential(
            nn.Linear(self.vit_dim, self.proj_dim),
            nn.LayerNorm(self.proj_dim),
            nn.ReLU()
        )
        
        # Multi-Head Attention layer
        # Stacks 3 tokens (ResNet, EffNet, ViT), each with projection dimension (512)
        num_heads = config["models"]["fusion"]["attention_heads"]
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.proj_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        # Fusion classification head
        dropout_rate = config["models"]["fusion"]["dropout"]
        dense_dim_1 = config["models"]["fusion"]["dense_dim_1"]  # 512
        dense_dim_2 = config["models"]["fusion"]["dense_dim_2"]  # 256
        
        self.classifier = nn.Sequential(
            nn.Linear(self.proj_dim * 3, dense_dim_1),
            nn.BatchNorm1d(dense_dim_1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(dense_dim_1, dense_dim_2),
            nn.BatchNorm1d(dense_dim_2),
            nn.ReLU(),
            nn.Linear(dense_dim_2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, resnet_feats: torch.Tensor, effnet_feats: torch.Tensor, vit_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the fusion network.
        
        Args:
            resnet_feats: Tensor of shape (B, 2048)
            effnet_feats: Tensor of shape (B, 1536)
            vit_feats: Tensor of shape (B, 768)
            
        Returns:
            A tuple of (predictions, attention_weights).
            - predictions: Tensor of shape (B, 1) containing tumor probabilities.
            - attention_weights: Tensor of shape (B, 3, 3) containing attention map.
        """
        # 1. Project feature vectors to a common 512 dimension
        proj_resnet = self.resnet_proj(resnet_feats)  # (B, 512)
        proj_effnet = self.effnet_proj(effnet_feats)  # (B, 512)
        proj_vit = self.vit_proj(vit_feats)            # (B, 512)
        
        # 2. Stack tokens to shape (B, 3, 512)
        stacked_features = torch.stack([proj_resnet, proj_effnet, proj_vit], dim=1)
        
        # 3. Apply Multihead Self-Attention
        # attn_output: (B, 3, 512), attn_weights: (B, 3, 3)
        attn_output, attn_weights = self.self_attention(
            stacked_features, 
            stacked_features, 
            stacked_features
        )
        
        # 4. Concatenate (flatten) the sequence of attended tokens to shape (B, 1536)
        flattened_attn = attn_output.reshape(attn_output.size(0), -1)
        
        # 5. Classification head outputs tumor probability (B, 1)
        probs = self.classifier(flattened_attn)
        
        return probs, attn_weights
