import os
import random
import numpy as np
import cv2
import yaml

def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def create_synthetic_mri(has_tumor: bool, img_size: int = 224) -> np.ndarray:
    """Generates a synthetic brain MRI image.
    
    Args:
        has_tumor: Whether to implant a tumor in the brain structure.
        img_size: Dimension of the output square image.
        
    Returns:
        A grayscale numpy array representing the simulated MRI slice (uint8).
    """
    # Initialize dark background
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    
    # 1. Draw skull boundary (outer ellipse)
    center = (img_size // 2, img_size // 2)
    axes = (int(img_size * 0.38), int(img_size * 0.45))
    cv2.ellipse(img, center, axes, 0, 0, 360, 40, -1)  # Outer skin/skull
    cv2.ellipse(img, center, (axes[0]-5, axes[1]-5), 0, 0, 360, 15, -1)  # Skull bone (darker gap)
    cv2.ellipse(img, center, (axes[0]-10, axes[1]-10), 0, 0, 360, 75, -1)  # Brain matter
    
    # 2. Draw brain matter internal structures (complex ventricles)
    v_left = (center[0] - int(img_size * 0.08), center[1] - int(img_size * 0.05))
    v_right = (center[0] + int(img_size * 0.08), center[1] - int(img_size * 0.05))
    cv2.ellipse(img, v_left, (int(img_size * 0.04), int(img_size * 0.12)), -15, 0, 360, 30, -1)
    cv2.ellipse(img, v_right, (int(img_size * 0.04), int(img_size * 0.12)), 15, 0, 360, 30, -1)
    
    # Add subtle brain ridges/folds (sulci)
    for i in range(12):
        angle = i * 30
        rad = np.deg2rad(angle)
        pt1 = (int(center[0] + (axes[0]-10)*0.7*np.cos(rad)), int(center[1] + (axes[1]-10)*0.7*np.sin(rad)))
        pt2 = (int(center[0] + (axes[0]-10)*np.cos(rad)), int(center[1] + (axes[1]-10)*np.sin(rad)))
        cv2.line(img, pt1, pt2, 45, 2)
        
    # 3. Add noise and smoothing for realistic MRI texture
    noise = np.random.normal(0, 8, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    # 4. Implant Tumor if requested
    if has_tumor:
        # Determine randomized placement inside brain matter (avoiding absolute border)
        t_angle = random.uniform(0, 2 * np.pi)
        t_dist = random.uniform(0.1, 0.25) * img_size
        t_center = (
            int(center[0] + t_dist * np.cos(t_angle)),
            int(center[1] + t_dist * np.sin(t_angle))
        )
        t_radius = random.randint(int(img_size * 0.05), int(img_size * 0.1))
        
        # Create a tumor mask
        tumor_mask = np.zeros_like(img, dtype=np.float32)
        cv2.circle(tumor_mask, t_center, t_radius, 1.0, -1)
        # Blur the tumor mask to create fuzzy edges
        tumor_mask = cv2.GaussianBlur(tumor_mask, (15, 15), 0)
        
        # Create edema (outer dark halo around the tumor)
        edema_mask = np.zeros_like(img, dtype=np.float32)
        cv2.circle(edema_mask, t_center, int(t_radius * 1.5), 1.0, -1)
        edema_mask = cv2.GaussianBlur(edema_mask, (25, 25), 0)
        
        # Apply edema (darkens brain matter)
        img = np.clip(img * (1.0 - edema_mask * 0.4), 0, 255).astype(np.uint8)
        
        # Apply bright tumor mass (bright white/grey contrast enhancement)
        tumor_overlay = (tumor_mask * 150).astype(np.uint8)
        img = cv2.addWeighted(img, 1.0, tumor_overlay, 0.9, 0)
        
    return img

def main():
    config = load_config()
    dataset_dir = config["paths"]["dataset_dir"]
    
    # Define splits
    splits = ["train", "val", "test"]
    categories = ["normal", "tumor"]
    
    # Count of images per split per class
    counts = {
        "train": {"normal": 60, "tumor": 60},
        "val": {"normal": 15, "tumor": 15},
        "test": {"normal": 15, "tumor": 15}
    }
    
    print("Generating synthetic Brain MRI dataset...")
    for split in splits:
        for category in categories:
            dir_path = os.path.join(dataset_dir, split, category)
            os.makedirs(dir_path, exist_ok=True)
            
            has_tumor = (category == "tumor")
            num_images = counts[split][category]
            
            for idx in range(num_images):
                img = create_synthetic_mri(has_tumor=has_tumor, img_size=config["data"]["image_size"][0])
                file_path = os.path.join(dir_path, f"{category}_{idx:03d}.png")
                cv2.imwrite(file_path, img)
                
    print(f"Dataset successfully created in '{dataset_dir}'")
    print(f"Total training images: {counts['train']['normal'] + counts['train']['tumor']}")
    print(f"Total validation images: {counts['val']['normal'] + counts['val']['tumor']}")
    print(f"Total test images: {counts['test']['normal'] + counts['test']['tumor']}")

if __name__ == "__main__":
    main()
