"""
Advanced preprocessing module for Brain MRI analysis.
Includes CLAHE normalization, augmentation, and dataset loading.

Key improvements:
- CLAHE for medical image enhancement
- Multiple augmentation strategies
- Deterministic preprocessing pipeline
- Compatible with PyTorch DataLoader
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import Dict, Any, Tuple, Optional

# Optional: Advanced augmentation (install albumentations for production)
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

class MRIPreprocessor:
    """Handles clinical preprocessing steps for Brain MRI scans."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.image_size = tuple(config["data"]["image_size"])
        self.clahe_clip = config["preprocessing"]["clahe_clip_limit"]
        self.clahe_grid = tuple(config["preprocessing"]["clahe_tile_grid_size"])
        self.blur_kernel = tuple(config["preprocessing"]["gaussian_kernel_size"])
        self.blur_sigma = config["preprocessing"]["gaussian_sigma"]
        self.enable_skull_strip = config["preprocessing"]["enable_skull_strip"]
        # Note: cv2.CLAHE is NOT picklable — do NOT store it as self.clahe.
        # It is created lazily inside apply_clahe() to support DataLoader multi-workers.
        
    def gaussian_blur(self, img: np.ndarray) -> np.ndarray:
        """Applies Gaussian Blur to smooth noise."""
        return cv2.GaussianBlur(img, self.blur_kernel, self.blur_sigma)
        
    def apply_clahe(self, img: np.ndarray) -> np.ndarray:
        """Applies Contrast Limited Adaptive Histogram Equalization.
        
        cv2.CLAHE objects are not picklable, so we create one fresh each call
        rather than storing it as an instance variable.
        """
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=self.clahe_grid)
        return clahe.apply(img)
        
    def skull_strip(self, img: np.ndarray) -> np.ndarray:
        """Strips the skull from the brain scan using Otsu thresholding and contours."""
        # 1. Blur to smooth out details
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        
        # 2. Thresholding via Otsu's binarization
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 3. Morphological closing to fill gaps inside the brain region
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # 4. Find the largest contour (assumed to be the brain)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(img)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            # Fill the largest contour to make the mask
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)
            
            # Apply morphological erosion to pull back the boundary slightly (skips outer skin/skull)
            mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            
            # Mask the original image
            stripped = cv2.bitwise_and(img, img, mask=mask)
            return stripped
        return img

    def preprocess_image(self, img_path: str) -> np.ndarray:
        """Applies the full preprocessing pipeline on a single image path.
        
        Steps: Read -> Grayscale -> Blur -> CLAHE -> Skull Strip -> Resize -> RGB.
        """
        # Read image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {img_path}")
            
        # Apply pipeline
        img_processed = self.gaussian_blur(img)
        img_processed = self.apply_clahe(img_processed)
        
        if self.enable_skull_strip:
            img_processed = self.skull_strip(img_processed)
            
        # Resize to final dimensions
        img_processed = cv2.resize(img_processed, self.image_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert grayscale to 3-channel RGB
        img_rgb = cv2.cvtColor(img_processed, cv2.COLOR_GRAY2RGB)
        return img_rgb


class MRIDataset(Dataset):
    """PyTorch Dataset for Brain MRI classification, with data augmentation."""
    
    def __init__(self, dataset_dir: str, split: str, config: Dict[str, Any], augment: bool = False):
        self.config = config
        self.split = split
        self.augment = augment
        self.preprocessor = MRIPreprocessor(config)
        
        self.image_paths = []
        self.labels = []
        
        # Binary class mapping: all tumor types = 1, no tumor = 0
        # Supports both old (tumor/normal) and Kaggle (glioma/meningioma/pituitary/notumor) formats
        self.tumor_classes = {"tumor", "glioma", "meningioma", "pituitary"}
        self.normal_classes = {"normal", "notumor", "no_tumor"}
        
        # Try multiple possible split folder names (handles Training/Testing and train/test)
        split_aliases = {
            "train": ["train", "Training"],
            "val":   ["val"],
            "test":  ["test", "Testing"],
        }
        split_dir = None
        for alias in split_aliases.get(split, [split]):
            candidate = os.path.join(dataset_dir, alias)
            if os.path.isdir(candidate):
                split_dir = candidate
                break
        
        if split_dir is None:
            raise FileNotFoundError(
                f"Could not find split '{split}' in {dataset_dir}. "
                f"Tried: {split_aliases.get(split, [split])}"
            )
            
        # Gather all images from all class subfolders
        for category in os.listdir(split_dir):
            cat_dir = os.path.join(split_dir, category)
            if not os.path.isdir(cat_dir):
                continue
            cat_lower = category.lower()
            if cat_lower in self.tumor_classes:
                label = 1
            elif cat_lower in self.normal_classes:
                label = 0
            else:
                continue  # skip unknown folders
            for fname in os.listdir(cat_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(cat_dir, fname))
                    self.labels.append(label)
                        
        # Setup transformations
        self.normalizer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Set up data augmentation
        aug_cfg = config["preprocessing"]["augmentation"]
        self.aug_transforms = transforms.Compose([
            transforms.RandomRotation(degrees=aug_cfg["rotation_range"]),
            transforms.RandomResizedCrop(
                size=tuple(config["data"]["image_size"]),
                scale=tuple(aug_cfg["zoom_range"]),
                ratio=(1.0, 1.0)
            ),
            transforms.ColorJitter(
                brightness=aug_cfg["brightness_range"][1] - 1.0,
                contrast=0.1
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=tuple(aug_cfg["translation_range"])
            ),
            transforms.RandomHorizontalFlip(p=0.5 if aug_cfg["horizontal_flip"] else 0.0),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 1. Preprocess using OpenCV pipeline
        preprocessed_img = self.preprocessor.preprocess_image(img_path)
        
        # Convert to PIL Image for torchvision transforms
        pil_img = Image.fromarray(preprocessed_img)
        
        # 2. Augment if training and requested
        if self.augment:
            pil_img = self.aug_transforms(pil_img)
            
        # 3. Normalize & Tensor conversion
        tensor_img = self.normalizer(pil_img)
        
        return tensor_img, torch.tensor(label, dtype=torch.float32)
