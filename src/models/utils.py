"""
Training utilities module for Brain Tumor Ensemble project.
Provides utilities for reproducibility, cross-validation, and advanced training techniques.
"""

import os
import random
import numpy as np
import torch
from typing import List, Tuple
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold


class EarlyStopping:
    """
    Early stopping callback for training.
    
    Stops training if validation loss doesn't improve for a specified number of epochs.
    
    Why use early stopping:
    - Prevents overfitting
    - Saves training time
    - Ensures best model is used
    - Critical for deep learning projects
    """
    
    def __init__(self, patience: int = 7, delta: float = 0.0, verbose: bool = True):
        """
        Args:
            patience: Number of epochs with no improvement after which training stops
            delta: Minimum change in the monitored value to qualify as improvement
            verbose: Whether to print messages
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """
        Check if early stopping criteria met.
        
        Args:
            val_loss: Current validation loss
        
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss >= self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


def set_deterministic_training(seed: int = 42):
    """
    Set all random seeds to ensure reproducible results across runs.
    
    Args:
        seed: Random seed value (default: 42)
    
    Why this matters:
    - Ensures reproducible results across different runs
    - Important for research and publication
    - Helps in debugging and comparing experiments
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior in CUDA operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"[INFO] Deterministic training enabled with seed: {seed}")


def get_kfold_splits(dataset, n_splits: int = 5, seed: int = 42) -> List[Tuple[List[int], List[int]]]:
    """
    Generate K-Fold cross-validation splits for a dataset.
    
    Args:
        dataset: PyTorch Dataset object
        n_splits: Number of folds (default: 5)
        seed: Random seed for reproducibility
    
    Returns:
        List of (train_indices, val_indices) tuples for each fold
    
    Why 5-Fold CV:
    - Better utilization of limited data
    - More robust performance estimation
    - Reduces variance in model evaluation
    - Standard practice in medical imaging research
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    indices = np.arange(len(dataset))
    
    splits = []
    for train_idx, val_idx in kfold.split(indices):
        splits.append((train_idx.tolist(), val_idx.tolist()))
    
    print(f"[INFO] Generated {n_splits}-Fold splits with {len(dataset)} samples")
    return splits


def create_fold_dataloaders(
    dataset,
    fold_indices: Tuple[List[int], List[int]],
    batch_size: int = 16,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for a specific fold.
    
    Args:
        dataset: Full dataset
        fold_indices: Tuple of (train_indices, val_indices)
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
    
    Returns:
        Tuple of (train_loader, val_loader)
    
    Why separate dataloaders per fold:
    - Ensures proper data separation between folds
    - Prevents data leakage
    - Each fold sees different validation data
    """
    train_idx, val_idx = fold_indices
    
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"[INFO] Fold split - Train: {len(train_idx)}, Val: {len(val_idx)}")
    return train_loader, val_loader


class TestTimeAugmentation:
    """
    Test Time Augmentation (TTA) for improved inference robustness.
    
    Applies multiple augmentations during inference and averages predictions.
    This improves model robustness and reduces prediction variance.
    
    Why TTA:
    - Reduces prediction uncertainty
    - Makes model more robust to variations
    - Slight computational overhead during inference
    - Common practice in competitive ML and medical imaging
    """
    
    def __init__(self, num_augmentations: int = 4):
        """
        Args:
            num_augmentations: Number of TTA augmentations (default: 4)
                - Original + 3 augmented versions = 4 predictions averaged
        """
        self.num_augmentations = num_augmentations
    
    def apply_tta(
        self,
        model: torch.nn.Module,
        image: torch.Tensor,
        device: torch.device,
        return_std: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply test-time augmentation to an image and return averaged prediction.
        
        Args:
            model: Neural network model in eval mode
            image: Input image tensor (B, C, H, W)
            device: Torch device
            return_std: Whether to return standard deviation of predictions
        
        Returns:
            Tuple of (averaged_predictions, std_of_predictions or all_predictions)
        
        Augmentation strategy:
        1. Original image (no augmentation)
        2. Horizontal flip
        3. Vertical flip
        4. Random rotation
        """
        predictions = []
        
        with torch.no_grad():
            # 1. Original prediction
            logits, features = model(image.to(device))
            probs = torch.sigmoid(logits)
            predictions.append(probs)
            
            if self.num_augmentations > 1:
                # 2. Horizontal flip
                h_flipped = torch.flip(image, dims=[-1])
                logits, _ = model(h_flipped.to(device))
                probs = torch.sigmoid(logits)
                predictions.append(probs)
            
            if self.num_augmentations > 2:
                # 3. Vertical flip
                v_flipped = torch.flip(image, dims=[-2])
                logits, _ = model(v_flipped.to(device))
                probs = torch.sigmoid(logits)
                predictions.append(probs)
            
            if self.num_augmentations > 3:
                # 4. Random rotation-like effect (for 2D, simulate via transpose/flip combo)
                rot90 = torch.rot90(image, k=1, dims=[-2, -1])
                logits, _ = model(rot90.to(device))
                probs = torch.sigmoid(logits)
                predictions.append(probs)
        
        predictions = torch.cat(predictions, dim=0)
        mean_pred = predictions.mean(dim=0, keepdim=True)
        
        if return_std:
            std_pred = predictions.std(dim=0, keepdim=True)
            return mean_pred, std_pred
        else:
            return mean_pred, predictions
    
    @staticmethod
    def average_predictions(predictions_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Average predictions from multiple models or augmentations.
        
        Args:
            predictions_list: List of prediction tensors
        
        Returns:
            Averaged predictions tensor
        
        Why averaging:
        - Reduces model variance
        - Better calibrated predictions
        - Improves generalization
        """
        stacked = torch.stack(predictions_list)
        return stacked.mean(dim=0)


class FocalLoss(torch.nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Focal Loss reduces the contribution of easy negative examples during training,
    focusing on harder examples. This is particularly useful for medical imaging
    where positive cases (tumors) are often rare.
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (RetinaNet)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter (higher = more focus on hard examples)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Model predictions (B, 1)
            targets: Ground truth labels (B, 1)
        
        Returns:
            Focal loss value
        """
        # Binary Cross Entropy
        bce_loss = torch.nn.functional.binary_cross_entropy(predictions, targets, reduction='none')
        
        # Focal term: (1 - p_t)^gamma
        p_t = predictions * targets + (1 - predictions) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Focal loss = -alpha * focal_weight * log(p_t)
        focal_loss = self.alpha * focal_weight * bce_loss
        
        return focal_loss.mean()


class MetricsTracker:
    """
    Tracks training metrics across epochs and folds.
    
    Maintains running statistics for:
    - Training and validation losses
    - Accuracy, Precision, Recall, F1, AUC
    - Per-fold and overall statistics
    """
    
    def __init__(self):
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'train_f1': [],
            'val_f1': [],
            'learning_rates': [],
            'fold_results': []
        }
    
    def add_epoch_metrics(self, epoch_dict: dict):
        """Add metrics for an epoch."""
        for key, value in epoch_dict.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def add_fold_result(self, fold_num: int, fold_dict: dict):
        """Record final metrics for a fold."""
        fold_dict['fold'] = fold_num
        self.metrics['fold_results'].append(fold_dict)
    
    def get_fold_average(self, metric_name: str) -> float:
        """Get average metric across all folds."""
        values = [f[metric_name] for f in self.metrics['fold_results'] if metric_name in f]
        return np.mean(values) if values else 0.0
    
    def get_fold_std(self, metric_name: str) -> float:
        """Get standard deviation of metric across folds."""
        values = [f[metric_name] for f in self.metrics['fold_results'] if metric_name in f]
        return np.std(values) if values else 0.0


def create_checkpoint_dir(base_dir: str = "checkpoints") -> str:
    """Create and return checkpoint directory path."""
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def save_best_model(model: torch.nn.Module, metrics: dict, checkpoint_path: str):
    """Save model checkpoint with metadata."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'timestamp': os.path.basename(checkpoint_path).split('.')[0]
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"[INFO] Saved best model checkpoint to {checkpoint_path}")


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"[INFO] Loaded model from {checkpoint_path}")
        return checkpoint.get('metrics', {}) if isinstance(checkpoint, dict) else {}
    else:
        print(f"[WARNING] Checkpoint not found: {checkpoint_path}")
        return {}
    """
    Set all random seeds to ensure reproducible results across runs.
    
    Args:
        seed: Random seed value (default: 42)
    
    Why this matters:
    - Ensures reproducible results across different runs
    - Important for research and publication
    - Helps in debugging and comparing experiments
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior in CUDA operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"[INFO] Deterministic training enabled with seed: {seed}")


def get_kfold_splits(dataset, n_splits: int = 5, seed: int = 42) -> List[Tuple[List[int], List[int]]]:
    """
    Generate K-Fold cross-validation splits for a dataset.
    
    Args:
        dataset: PyTorch Dataset object
        n_splits: Number of folds (default: 5)
        seed: Random seed for reproducibility
    
    Returns:
        List of (train_indices, val_indices) tuples for each fold
    
    Why 5-Fold CV:
    - Better utilization of limited data
    - More robust performance estimation
    - Reduces variance in model evaluation
    - Standard practice in medical imaging research
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    indices = np.arange(len(dataset))
    
    splits = []
    for train_idx, val_idx in kfold.split(indices):
        splits.append((train_idx.tolist(), val_idx.tolist()))
    
    print(f"[INFO] Generated {n_splits}-Fold splits with {len(dataset)} samples")
    return splits


def create_fold_dataloaders(
    dataset,
    fold_indices: Tuple[List[int], List[int]],
    batch_size: int = 16,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for a specific fold.
    
    Args:
        dataset: Full dataset
        fold_indices: Tuple of (train_indices, val_indices)
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
    
    Returns:
        Tuple of (train_loader, val_loader)
    
    Why separate dataloaders per fold:
    - Ensures proper data separation between folds
    - Prevents data leakage
    - Each fold sees different validation data
    """
    train_idx, val_idx = fold_indices
    
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"[INFO] Fold split - Train: {len(train_idx)}, Val: {len(val_idx)}")
    return train_loader, val_loader


class TestTimeAugmentation:
    """
    Test Time Augmentation (TTA) for improved inference robustness.
    
    Applies multiple augmentations during inference and averages predictions.
    This improves model robustness and reduces prediction variance.
    
    Why TTA:
    - Reduces prediction uncertainty
    - Makes model more robust to variations
    - Slight computational overhead during inference
    - Common practice in competitive ML and medical imaging
    """
    
    def __init__(self, num_augmentations: int = 4):
        """
        Args:
            num_augmentations: Number of TTA augmentations (default: 4)
                - Original + 3 augmented versions = 4 predictions averaged
        """
        self.num_augmentations = num_augmentations
    
    def apply_tta(
        self,
        model: torch.nn.Module,
        image: torch.Tensor,
        device: torch.device,
        return_std: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply test-time augmentation to an image and return averaged prediction.
        
        Args:
            model: Neural network model in eval mode
            image: Input image tensor (B, C, H, W)
            device: Torch device
            return_std: Whether to return standard deviation of predictions
        
        Returns:
            Tuple of (averaged_predictions, std_of_predictions or all_predictions)
        
        Augmentation strategy:
        1. Original image (no augmentation)
        2. Horizontal flip
        3. Vertical flip
        4. Random rotation
        """
        predictions = []
        
        with torch.no_grad():
            # 1. Original prediction
            logits, features = model(image.to(device))
            probs = torch.sigmoid(logits)
            predictions.append(probs)
            
            if self.num_augmentations > 1:
                # 2. Horizontal flip
                h_flipped = torch.flip(image, dims=[-1])
                logits, _ = model(h_flipped.to(device))
                probs = torch.sigmoid(logits)
                predictions.append(probs)
            
            if self.num_augmentations > 2:
                # 3. Vertical flip
                v_flipped = torch.flip(image, dims=[-2])
                logits, _ = model(v_flipped.to(device))
                probs = torch.sigmoid(logits)
                predictions.append(probs)
            
            if self.num_augmentations > 3:
                # 4. Random rotation-like effect (for 2D, simulate via transpose/flip combo)
                rot90 = torch.rot90(image, k=1, dims=[-2, -1])
                logits, _ = model(rot90.to(device))
                probs = torch.sigmoid(logits)
                predictions.append(probs)
        
        predictions = torch.cat(predictions, dim=0)
        mean_pred = predictions.mean(dim=0, keepdim=True)
        
        if return_std:
            std_pred = predictions.std(dim=0, keepdim=True)
            return mean_pred, std_pred
        else:
            return mean_pred, predictions
    
    @staticmethod
    def average_predictions(predictions_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Average predictions from multiple models or augmentations.
        
        Args:
            predictions_list: List of prediction tensors
        
        Returns:
            Averaged predictions tensor
        
        Why averaging:
        - Reduces model variance
        - Better calibrated predictions
        - Improves generalization
        """
        stacked = torch.stack(predictions_list)
        return stacked.mean(dim=0)


class FocalLoss(torch.nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Focal Loss reduces the contribution of easy negative examples during training,
    focusing on harder examples. This is particularly useful for medical imaging
    where positive cases (tumors) are often rare.
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (RetinaNet)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter (higher = more focus on hard examples)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Model predictions (B, 1)
            targets: Ground truth labels (B, 1)
        
        Returns:
            Focal loss value
        """
        # Binary Cross Entropy
        bce_loss = torch.nn.functional.binary_cross_entropy(predictions, targets, reduction='none')
        
        # Focal term: (1 - p_t)^gamma
        p_t = predictions * targets + (1 - predictions) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Focal loss = -alpha * focal_weight * log(p_t)
        focal_loss = self.alpha * focal_weight * bce_loss
        
        return focal_loss.mean()


class MetricsTracker:
    """
    Tracks training metrics across epochs and folds.
    
    Maintains running statistics for:
    - Training and validation losses
    - Accuracy, Precision, Recall, F1, AUC
    - Per-fold and overall statistics
    """
    
    def __init__(self):
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'train_f1': [],
            'val_f1': [],
            'learning_rates': [],
            'fold_results': []
        }
    
    def add_epoch_metrics(self, epoch_dict: dict):
        """Add metrics for an epoch."""
        for key, value in epoch_dict.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def add_fold_result(self, fold_num: int, fold_dict: dict):
        """Record final metrics for a fold."""
        fold_dict['fold'] = fold_num
        self.metrics['fold_results'].append(fold_dict)
    
    def get_fold_average(self, metric_name: str) -> float:
        """Get average metric across all folds."""
        values = [f[metric_name] for f in self.metrics['fold_results'] if metric_name in f]
        return np.mean(values) if values else 0.0
    
    def get_fold_std(self, metric_name: str) -> float:
        """Get standard deviation of metric across folds."""
        values = [f[metric_name] for f in self.metrics['fold_results'] if metric_name in f]
        return np.std(values) if values else 0.0


def create_checkpoint_dir(base_dir: str = "checkpoints") -> str:
    """Create and return checkpoint directory path."""
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def save_best_model(model: torch.nn.Module, metrics: dict, checkpoint_path: str):
    """Save model checkpoint with metadata."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'timestamp': os.path.basename(checkpoint_path).split('.')[0]
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"[INFO] Saved best model checkpoint to {checkpoint_path}")


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"[INFO] Loaded model from {checkpoint_path}")
        return checkpoint.get('metrics', {}) if isinstance(checkpoint, dict) else {}
    else:
        print(f"[WARNING] Checkpoint not found: {checkpoint_path}")
        return {}
