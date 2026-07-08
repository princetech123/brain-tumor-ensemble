import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Tuple, Optional

class GradCAM:
    """Computes Grad-CAM visualizations for convolutional neural networks."""
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        
    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Generates a raw 2D Grad-CAM heatmap for a single input image.
        
        Args:
            input_tensor: Shape (1, 3, H, W)
            
        Returns:
            Grayscale heatmap of shape (H_target, W_target) normalized to [0, 1].
        """
        # Storage for hooks
        activations = []
        gradients = []
        
        def forward_hook(module, input, output):
            activations.append(output.detach())
            
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0].detach())
            
        # Register hooks dynamically to prevent leakage
        h_forward = self.target_layer.register_forward_hook(forward_hook)
        h_backward = self.target_layer.register_full_backward_hook(backward_hook)
        
        # Forward pass
        self.model.zero_grad()
        # Ensure gradients can be computed on input if required, but we backward from logit
        logits, _ = self.model(input_tensor)
        
        # We target the probability score of binary classification (Sigmoid/Logits)
        # If logits is shape (1, 1), score is logits[0, 0]
        # We backpropagate the raw logit (before sigmoid if standalone, or class score)
        score = logits[0, 0]
        score.backward()
        
        # Remove hooks immediately
        h_forward.remove()
        h_backward.remove()
        
        # If no gradients or activations were captured
        if not activations or not gradients:
            raise RuntimeError("Grad-CAM hooks failed to capture activations or gradients. Check target layer.")
            
        act = activations[0]   # Shape: (1, Channels, H_feat, W_feat)
        grad = gradients[0]   # Shape: (1, Channels, H_feat, W_feat)
        
        # Channel-wise pooling of gradients
        weights = torch.mean(grad, dim=(2, 3), keepdim=True)  # (1, Channels, 1, 1)
        
        # Weighted sum of activations
        cam = torch.sum(weights * act, dim=1).squeeze(0)  # (H_feat, W_feat)
        
        # ReLU to keep only positive contributions (features that positively impact tumor prediction)
        cam = torch.clamp(cam, min=0)
        
        # Min-max normalization
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        else:
            cam = torch.zeros_like(cam)
            
        return cam.cpu().numpy()


class ViTAttentionMap:
    """Extracts attention map visualization from Vision Transformer (ViT-B/16)."""
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        
    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Generates a raw 2D Attention Map for ViT Class Token.
        
        Args:
            input_tensor: Shape (1, 3, H, W)
            
        Returns:
            Normalized attention map of shape (14, 14).
        """
        # Hook to capture input to the target transformer layer (attention block)
        inputs_list = []
        
        def forward_hook(module, input, output):
            # input[0] is the input tensor of shape (Batch, SeqLen, EmbedDim)
            inputs_list.append(input[0].detach())
            
        h_forward = self.target_layer.register_forward_hook(forward_hook)
        
        # Run forward pass (no backward needed)
        with torch.no_grad():
            _ = self.model(input_tensor)
            
        h_forward.remove()
        
        if not inputs_list:
            raise RuntimeError("ViT Attention hook failed to capture inputs.")
            
        x = inputs_list[0]  # Shape: (1, 197, 768) -> (Batch, SeqLen, Dim)
        
        # Split CLS token and patch tokens
        cls_token = x[:, 0, :]    # Shape: (1, 768)
        patch_tokens = x[:, 1:, :] # Shape: (1, 196, 768)
        
        # Compute dot-product cosine similarity between CLS token and all patch tokens
        # cls_token normalized: (1, 768)
        cls_norm = cls_token / (torch.norm(cls_token, dim=-1, keepdim=True) + 1e-8)
        # patch_tokens normalized: (1, 196, 768)
        patch_norm = patch_tokens / (torch.norm(patch_tokens, dim=-1, keepdim=True) + 1e-8)
        
        # Dot product: (1, 196)
        similarity = torch.matmul(cls_norm.unsqueeze(1), patch_norm.transpose(-1, -2)).squeeze(0).squeeze(0)
        
        # Reshape sequence of 196 tokens into 14x14 grid
        attn_map = similarity.reshape(14, 14)
        
        # Apply min-max normalization
        map_min, map_max = attn_map.min(), attn_map.max()
        if map_max > map_min:
            attn_map = (attn_map - map_min) / (map_max - map_min + 1e-8)
        else:
            attn_map = torch.zeros_like(attn_map)
            
        return attn_map.cpu().numpy()


def overlay_heatmap(img_rgb: np.ndarray, heatmap_raw: np.ndarray, alpha: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Overlays the generated 2D heatmap onto the original image.
    
    Args:
        img_rgb: Original preprocessed image, shape (224, 224, 3), values [0, 255] uint8.
        heatmap_raw: 2D heatmap in [0, 1] range.
        alpha: Blending transparency coefficient.
        
    Returns:
        A tuple of (colorized_heatmap, overlayed_image).
        - colorized_heatmap: JET colormapped heatmap.
        - overlayed_image: Combined overlay image.
    """
    # 1. Resize heatmap to match image size (224, 224)
    heatmap_resized = cv2.resize(heatmap_raw, (img_rgb.shape[1], img_rgb.shape[0]))
    
    # 2. Convert to uint8 (0-255)
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    
    # 3. Apply color map
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # Convert JET color map (BGR in OpenCV) to RGB
    heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # 4. Blend images
    overlay = cv2.addWeighted(heatmap_color_rgb, alpha, img_rgb, 1.0 - alpha, 0)
    
    return heatmap_color_rgb, overlay
