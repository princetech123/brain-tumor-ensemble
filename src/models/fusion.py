import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

class EnsembleFusionModel(nn.Module):
    """
    Advanced Ensemble model that fuses feature vectors from ResNet50, EfficientNet-B3, and ViT-B/16
    using a Multi-Head Self-Attention mechanism with trainable weights.
    
    Features:
    - Multi-Head Self-Attention for dynamic weight allocation
    - Trainable per-backbone weights for explicit feature importance
    - Progressive feature fusion with layer normalization
    - Research-grade architecture for optimal performance
    
    Args:
        config: Configuration dictionary with model parameters
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Feature dimensions from configuration
        self.resnet_dim = config["models"]["resnet"]["feature_dim"]      # 2048
        self.effnet_dim = config["models"]["efficientnet"]["feature_dim"] # 1536
        self.vit_dim = config["models"]["vit"]["feature_dim"]             # 768
        
        # Shared attention projection dimension
        self.proj_dim = config["models"]["fusion"]["projection_dim"]  # 512
        
        # Projection layers to map all features to a common dimension
        # These help align feature spaces from different architectures
        self.resnet_proj = nn.Sequential(
            nn.Linear(self.resnet_dim, self.proj_dim),
            nn.LayerNorm(self.proj_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.effnet_proj = nn.Sequential(
            nn.Linear(self.effnet_dim, self.proj_dim),
            nn.LayerNorm(self.proj_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.vit_proj = nn.Sequential(
            nn.Linear(self.vit_dim, self.proj_dim),
            nn.LayerNorm(self.proj_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Trainable weights for each backbone
        # Allow the model to learn the importance of each backbone
        use_learnable_weights = config["models"]["fusion"]["use_learnable_weights"]
        if use_learnable_weights:
            self.resnet_weight = nn.Parameter(torch.ones(1) * 0.33)
            self.effnet_weight = nn.Parameter(torch.ones(1) * 0.33)
            self.vit_weight = nn.Parameter(torch.ones(1) * 0.34)
            self.register_parameter('resnet_weight', self.resnet_weight)
            self.register_parameter('effnet_weight', self.effnet_weight)
            self.register_parameter('vit_weight', self.vit_weight)
        else:
            self.resnet_weight = None
            self.effnet_weight = None
            self.vit_weight = None
        
        # Multi-Head Attention layer
        # Stacks 3 tokens (ResNet, EffNet, ViT), each with dimension 512
        # This allows the model to learn complex interactions between backbones
        num_heads = config["models"]["fusion"]["attention_heads"]
        num_layers = config["models"]["fusion"]["attention_layers"]
        
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.proj_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # Additional transformer layer for deeper fusion
        if num_layers > 1:
            self.transformer_encoder = nn.TransformerEncoderLayer(
                d_model=self.proj_dim,
                nhead=num_heads,
                dim_feedforward=self.proj_dim * 4,
                dropout=0.1,
                batch_first=True,
                activation='relu'
            )
        else:
            self.transformer_encoder = None
        
        # Fusion classification head
        dropout_rate = config["models"]["fusion"]["dropout"]
        dense_dim_1 = config["models"]["fusion"]["dense_dim_1"]
        dense_dim_2 = config["models"]["fusion"]["dense_dim_2"]
        
        # Improved classifier with batch normalization and regularization
        # BACKWARD COMPATIBLE: Can load old checkpoints without weights/bias on layer 8
        self.classifier = nn.Sequential(
            nn.Linear(self.proj_dim * 3, dense_dim_1),      # Layer 0
            nn.BatchNorm1d(dense_dim_1),                    # Layer 1
            nn.ReLU(),                                      # Layer 2
            nn.Dropout(p=dropout_rate),                     # Layer 3
            nn.Linear(dense_dim_1, dense_dim_2),            # Layer 4
            nn.BatchNorm1d(dense_dim_2),                    # Layer 5
            nn.ReLU(),                                      # Layer 6
            nn.Dropout(p=dropout_rate),                     # Layer 7
            nn.Linear(dense_dim_2, 1),                      # Layer 8 (new, for backward compat)
            nn.Sigmoid()                                    # Layer 9
        )
    
    def load_state_dict(self, state_dict, strict=True):
        """
        Override load_state_dict to handle backward compatibility with old checkpoints.
        
        Old checkpoints have:
        - No trainable weights (resnet_weight, effnet_weight, vit_weight)
        - Classifier ending at layer 7 (classifier.7.*)
        
        New checkpoints have:
        - Trainable weights
        - Classifier with 8 layers (classifier.8.*)
        """
        # Handle missing trainable weights (old checkpoints)
        if 'resnet_weight' not in state_dict and self.resnet_weight is not None:
            print("[WARNING] Old checkpoint detected - initializing trainable weights")
            # Initialize with default values
            state_dict['resnet_weight'] = torch.ones(1) * 0.33
            state_dict['effnet_weight'] = torch.ones(1) * 0.33
            state_dict['vit_weight'] = torch.ones(1) * 0.34
        
        # Handle old classifier that ends at layer 7 instead of layer 8
        old_classifier_keys = []
        new_classifier_keys = []
        for key in list(state_dict.keys()):
            if key.startswith('classifier.7.'):
                old_classifier_keys.append(key)
            elif key.startswith('classifier.8.'):
                new_classifier_keys.append(key)
        
        if old_classifier_keys and not new_classifier_keys:
            print("[WARNING] Old checkpoint classifier format detected - converting")
            # Old checkpoint ends at layer 7 (last linear layer without Sigmoid)
            # We need to convert this to new format with layer 8
            # Copy old layer 7 weights to new layer 8
            for key in old_classifier_keys:
                if 'classifier.7.' in key:
                    new_key = key.replace('classifier.7.', 'classifier.8.')
                    state_dict[new_key] = state_dict.pop(key)
        
        # Use non-strict loading if we have incompatibilities
        if not strict or ('resnet_weight' not in state_dict):
            print("[INFO] Using non-strict loading for checkpoint compatibility")
            return super().load_state_dict(state_dict, strict=False)
        else:
            return super().load_state_dict(state_dict, strict=strict)
        
    def forward(
        self,
        resnet_feats: torch.Tensor,
        effnet_feats: torch.Tensor,
        vit_feats: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the fusion network.
        
        Args:
            resnet_feats: Tensor of shape (B, 2048) from ResNet50
            effnet_feats: Tensor of shape (B, 1536) from EfficientNet-B3
            vit_feats: Tensor of shape (B, 768) from ViT-B/16
            
        Returns:
            Tuple of (predictions, attention_weights):
            - predictions: Tensor of shape (B, 1) containing tumor probabilities [0-1]
            - attention_weights: Tensor of shape (B, 3, 3) containing attention weights
                                showing how much each backbone attends to others
        
        Process:
        1. Project features to common dimension (512)
        2. Apply trainable weights if enabled
        3. Stack projections and apply multi-head attention
        4. Optional: Apply transformer encoder for deeper fusion
        5. Flatten and classify
        """
        # 1. Project feature vectors to common dimension (512)
        proj_resnet = self.resnet_proj(resnet_feats)   # (B, 512)
        proj_effnet = self.effnet_proj(effnet_feats)   # (B, 512)
        proj_vit = self.vit_proj(vit_feats)             # (B, 512)
        
        # 2. Apply trainable weights (if enabled)
        # Allows model to learn backbone importance
        if self.resnet_weight is not None:
            # Normalize weights to sum to 1 using softmax
            weights = torch.softmax(
                torch.stack([self.resnet_weight, self.effnet_weight, self.vit_weight]),
                dim=0
            )
            proj_resnet = proj_resnet * weights[0]
            proj_effnet = proj_effnet * weights[1]
            proj_vit = proj_vit * weights[2]
        
        # 3. Stack tokens to shape (B, 3, 512)
        stacked_features = torch.stack([proj_resnet, proj_effnet, proj_vit], dim=1)
        
        # 4. Apply Multi-Head Self-Attention
        # attn_output: (B, 3, 512) - attended features
        # attn_weights: (B, num_heads, 3, 3) - attention scores
        attn_output, attn_weights = self.self_attention(
            stacked_features,
            stacked_features,
            stacked_features
        )
        
        # 5. Optional: Apply transformer encoder for deeper fusion
        if self.transformer_encoder is not None:
            attn_output = self.transformer_encoder(attn_output)
        
        # 6. Concatenate (flatten) the sequence of attended tokens
        # Shape: (B, 512 * 3) = (B, 1536)
        flattened_attn = attn_output.reshape(attn_output.size(0), -1)
        
        # 7. Classification head outputs tumor probability
        # Shape: (B, 1), range [0, 1]
        probs = self.classifier(flattened_attn)
        
        return probs, attn_weights

