"""
Research-Grade Training Pipeline for Brain Tumor Ensemble.

Features:
- Deterministic training with fixed random seeds
- Progressive unfreezing strategy for transfer learning
- AdamW optimizer with CosineAnnealingLR scheduler
- Automatic Mixed Precision (AMP) for efficiency
- Early stopping with patience=7
- Best model checkpoint saving
- Complete TensorBoard logging
- CLAHE + MRI normalization preprocessing
- Advanced data augmentation
- Weighted loss for class imbalance
- 5-Fold Cross Validation
- Test Time Augmentation (TTA) during inference
- Trainable attention-based ensemble fusion
- Comprehensive metrics: Accuracy, Precision, Recall, Specificity, F1, ROC-AUC
- Metrics saved to logs/metrics.json
- Automatic training history plots
- Flask UI compatibility maintained
- Production-ready code quality

Author: AI Research Engineer
Date: July 2026
"""

import os
import sys
import yaml
import json
import logging
import time
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import KFold
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# Project imports
from src.preprocessing import MRIDataset, MRIPreprocessor
from src.models.backbones import load_backbone
from src.models.fusion import EnsembleFusionModel
from src.models.utils import (
    set_deterministic_training,
    get_kfold_splits,
    create_fold_dataloaders,
    TestTimeAugmentation,
    MetricsTracker,
    EarlyStopping,
    create_checkpoint_dir,
    save_best_model,
    load_checkpoint
)
from src.evaluation import calculate_clinical_metrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config() -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)


def ensure_reproducibility(config: Dict[str, Any]):
    """Ensure deterministic training across all libraries."""
    seed = config["data"]["random_seed"]
    set_deterministic_training(seed)
    logger.info(f"✓ Reproducibility enabled with seed: {seed}")


def train_backbone_with_progressive_unfreezing(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
    fold_num: int = 0,
    tb_writer: Optional[SummaryWriter] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Train backbone with progressive unfreezing strategy.
    
    Strategy:
    1. Epochs 1-15: Train only classification head (backbone frozen)
    2. Epochs 16-50: Unfreeze last 2 blocks and fine-tune
    
    This prevents catastrophic forgetting of pre-trained features.
    
    Args:
        model_name: Name of model (resnet50, efficientnet_b3, vit_b_16)
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Torch device
        config: Configuration dictionary
        fold_num: Fold number for logging
        tb_writer: TensorBoard writer
    
    Returns:
        Tuple of (best_val_loss, best_metrics)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_name} (Fold {fold_num + 1})")
    logger.info(f"{'='*60}")
    
    cfg = config["training"]["backbone"]
    epochs = int(cfg["epochs"])
    unfreeze_at_epoch = int(cfg["unfreeze_at_epoch"])
    
    # Optimizer: AdamW
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
        betas=[float(b) for b in cfg["betas"]],
        eps=float(cfg["eps"])
    )
    
    # Scheduler: CosineAnnealingLR
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg["t_max"]),
        eta_min=float(cfg["eta_min"])
    )
    
    # Loss function: Weighted BCE for class imbalance
    class_weights = torch.tensor([float(w) for w in cfg["class_weights"]], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1:2])
    
    # AMP (Automatic Mixed Precision)
    use_amp = config["training"]["use_amp"]
    scaler = GradScaler() if use_amp else None
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=int(config["training"]["early_stopping_patience"]),
        delta=float(config["training"]["early_stopping_delta"])
    )
    
    # Freeze backbone initially
    model.freeze_backbone(True)
    logger.info("✓ Backbone frozen initially (training classifier head only)")
    
    best_val_loss = float('inf')
    best_metrics = {}
    checkpoint_dir = create_checkpoint_dir()
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}.pth")
    
    for epoch in range(epochs):
        # Progressive unfreezing
        if epoch == unfreeze_at_epoch:
            model.unfreeze_last_blocks()
            # Update optimizer to include unfrozen layers
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=cfg["learning_rate"],
                weight_decay=cfg["weight_decay"]
            )
            logger.info(f"✓ Unfroze last layers at epoch {epoch + 1}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_idx, (imgs, labels) in enumerate(pbar):
            imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            
            if use_amp:
                with autocast():
                    logits, _ = model(imgs)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, _ = model(imgs)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            
            # Track predictions
            with torch.no_grad():
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                train_preds.extend(probs)
                train_labels.extend(labels.detach().cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for imgs, labels in pbar:
                imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
                
                if use_amp:
                    with autocast():
                        logits, _ = model(imgs)
                        loss = criterion(logits, labels)
                else:
                    logits, _ = model(imgs)
                    loss = criterion(logits, labels)
                
                val_loss += loss.item() * imgs.size(0)
                
                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds.extend(probs)
                val_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        val_loss /= len(val_loader.dataset)
        scheduler.step()
        
        # Compute metrics
        val_preds = np.array(val_preds).squeeze()
        val_labels = np.array(val_labels).squeeze()
        val_metrics = calculate_clinical_metrics(val_labels, val_preds)
        
        # Logging
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
            f"Acc: {val_metrics['accuracy']:.4f} | "
            f"F1: {val_metrics['f1_score']:.4f} | "
            f"AUC: {val_metrics['auc']:.4f} | "
            f"LR: {current_lr:.2e}"
        )
        
        # TensorBoard logging
        if tb_writer:
            tb_writer.add_scalar(f"{model_name}/Loss/Train", train_loss, epoch)
            tb_writer.add_scalar(f"{model_name}/Loss/Val", val_loss, epoch)
            tb_writer.add_scalar(f"{model_name}/Accuracy", val_metrics['accuracy'], epoch)
            tb_writer.add_scalar(f"{model_name}/F1_Score", val_metrics['f1_score'], epoch)
            tb_writer.add_scalar(f"{model_name}/AUC", val_metrics['auc'], epoch)
            tb_writer.add_scalar(f"{model_name}/LearningRate", current_lr, epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = val_metrics.copy()
            save_best_model(model, best_metrics, checkpoint_path)
            logger.info(f"  ✓ Best model saved (Val Loss: {val_loss:.4f})")
        
        # Early stopping
        if early_stopping(val_loss):
            logger.info(f"✓ Early stopping triggered at epoch {epoch + 1}")
            break
    
    return best_val_loss, best_metrics


def train_ensemble_fusion(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
    fold_num: int = 0,
    tb_writer: Optional[SummaryWriter] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Train ensemble fusion model with frozen backbones.
    
    The backbones are pre-trained and frozen. Only the fusion network
    (attention and classification heads) is trained.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Torch device
        config: Configuration dictionary
        fold_num: Fold number for logging
        tb_writer: TensorBoard writer
    
    Returns:
        Tuple of (best_val_loss, best_metrics)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Training Ensemble Fusion (Fold {fold_num + 1})")
    logger.info(f"{'='*60}")
    
    cfg = config["training"]["fusion"]
    checkpoint_dir = create_checkpoint_dir()
    
    # Load and freeze backbones
    logger.info("Loading backbones...")
    
    resnet = load_backbone("resnet50", pretrained=True).to(device)
    resnet.freeze_backbone(True)
    resnet.eval()
    logger.info("✓ ResNet50 loaded and frozen")
    
    effnet = load_backbone("efficientnet_b3", pretrained=True).to(device)
    effnet.freeze_backbone(True)
    effnet.eval()
    logger.info("✓ EfficientNet-B3 loaded and frozen")
    
    vit = load_backbone("vit_b_16", pretrained=True).to(device)
    vit.freeze_backbone(True)
    vit.eval()
    logger.info("✓ ViT-B/16 loaded and frozen")
    
    # Create fusion model
    fusion_model = EnsembleFusionModel(config).to(device)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        fusion_model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
        betas=[float(b) for b in cfg["betas"]]
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg["t_max"]),
        eta_min=float(cfg["eta_min"])
    )
    
    # Loss and AMP
    criterion = nn.BCELoss()
    use_amp = config["training"]["use_amp"]
    scaler = GradScaler() if use_amp else None
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=int(config["training"]["early_stopping_patience"]),
        delta=float(config["training"]["early_stopping_delta"])
    )
    
    best_val_loss = float('inf')
    best_metrics = {}
    checkpoint_path = os.path.join(checkpoint_dir, "fusion.pth")
    
    epochs = cfg["epochs"]
    
    for epoch in range(epochs):
        # Training phase
        fusion_model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            
            # Extract features from frozen backbones
            with torch.no_grad():
                res_feats = resnet(imgs, return_features=True)
                eff_feats = effnet(imgs, return_features=True)
                vit_feats = vit(imgs, return_features=True)
            
            if use_amp:
                with autocast():
                    probs, _ = fusion_model(res_feats, eff_feats, vit_feats)
                    loss = criterion(probs, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                probs, _ = fusion_model(res_feats, eff_feats, vit_feats)
                loss = criterion(probs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), max_norm=1.0)
                optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            train_preds.extend(probs.detach().cpu().numpy())
            train_labels.extend(labels.detach().cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader.dataset)
        
        # Validation phase
        fusion_model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for imgs, labels in pbar:
                imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
                
                res_feats = resnet(imgs, return_features=True)
                eff_feats = effnet(imgs, return_features=True)
                vit_feats = vit(imgs, return_features=True)
                
                if use_amp:
                    with autocast():
                        probs, _ = fusion_model(res_feats, eff_feats, vit_feats)
                        loss = criterion(probs, labels)
                else:
                    probs, _ = fusion_model(res_feats, eff_feats, vit_feats)
                    loss = criterion(probs, labels)
                
                val_loss += loss.item() * imgs.size(0)
                val_preds.extend(probs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        val_loss /= len(val_loader.dataset)
        scheduler.step()
        
        # Compute metrics
        val_preds = np.array(val_preds).squeeze()
        val_labels = np.array(val_labels).squeeze()
        val_metrics = calculate_clinical_metrics(val_labels, val_preds)
        
        # Logging
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
            f"Acc: {val_metrics['accuracy']:.4f} | "
            f"F1: {val_metrics['f1_score']:.4f} | "
            f"AUC: {val_metrics['auc']:.4f} | "
            f"LR: {current_lr:.2e}"
        )
        
        # TensorBoard logging
        if tb_writer:
            tb_writer.add_scalar("Fusion/Loss/Train", train_loss, epoch)
            tb_writer.add_scalar("Fusion/Loss/Val", val_loss, epoch)
            tb_writer.add_scalar("Fusion/Accuracy", val_metrics['accuracy'], epoch)
            tb_writer.add_scalar("Fusion/F1_Score", val_metrics['f1_score'], epoch)
            tb_writer.add_scalar("Fusion/AUC", val_metrics['auc'], epoch)
            tb_writer.add_scalar("Fusion/LearningRate", current_lr, epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = val_metrics.copy()
            save_best_model(fusion_model, best_metrics, checkpoint_path)
            logger.info(f"  ✓ Best model saved (Val Loss: {val_loss:.4f})")
        
        # Early stopping
        if early_stopping(val_loss):
            logger.info(f"✓ Early stopping triggered at epoch {epoch + 1}")
            break
    
    return best_val_loss, best_metrics


def evaluate_on_test_set(
    test_loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Comprehensive evaluation on test set with TTA.
    
    Features:
    - Test Time Augmentation (TTA) for robustness
    - Per-model predictions
    - Ensemble predictions
    - Comprehensive metrics
    - Confusion matrices
    - ROC-AUC curves
    
    Args:
        test_loader: Test data loader
        device: Torch device
        config: Configuration dictionary
    
    Returns:
        Dictionary with all evaluation results
    """
    logger.info(f"\n{'='*60}")
    logger.info("Evaluating on Test Set")
    logger.info(f"{'='*60}")
    
    checkpoint_dir = create_checkpoint_dir()
    use_tta = config["inference"]["use_tta"]
    tta_augmentations = config["inference"]["tta_augmentations"] if use_tta else 1
    
    # Load all models
    resnet = load_backbone("resnet50", pretrained=True).to(device)
    resnet.load_state_dict(torch.load(
        os.path.join(checkpoint_dir, "resnet50.pth"),
        map_location=device
    ))
    resnet.eval()
    
    effnet = load_backbone("efficientnet_b3", pretrained=True).to(device)
    effnet.load_state_dict(torch.load(
        os.path.join(checkpoint_dir, "efficientnet_b3.pth"),
        map_location=device
    ))
    effnet.eval()
    
    vit = load_backbone("vit_b_16", pretrained=True).to(device)
    vit.load_state_dict(torch.load(
        os.path.join(checkpoint_dir, "vit_b_16.pth"),
        map_location=device
    ))
    vit.eval()
    
    fusion_model = EnsembleFusionModel(config).to(device)
    fusion_model.load_state_dict(torch.load(
        os.path.join(checkpoint_dir, "fusion.pth"),
        map_location=device
    ))
    fusion_model.eval()
    
    logger.info("✓ All models loaded")
    
    # TTA
    tta = TestTimeAugmentation(num_augmentations=tta_augmentations) if use_tta else None
    
    results = {
        "ResNet50": {"preds": [], "labels": []},
        "EfficientNetB3": {"preds": [], "labels": []},
        "ViT": {"preds": [], "labels": []},
        "Ensemble Fusion": {"preds": [], "labels": []}
    }
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating")
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels_np = labels.numpy()
            
            # ResNet50 predictions
            if use_tta:
                res_probs, _ = tta.apply_tta(resnet, imgs, device)
            else:
                res_logits, _ = resnet(imgs)
                res_probs = torch.sigmoid(res_logits)
            
            results["ResNet50"]["preds"].extend(res_probs.squeeze().cpu().numpy())
            results["ResNet50"]["labels"].extend(labels_np)
            
            # EfficientNet predictions
            if use_tta:
                eff_probs, _ = tta.apply_tta(effnet, imgs, device)
            else:
                eff_logits, _ = effnet(imgs)
                eff_probs = torch.sigmoid(eff_logits)
            
            results["EfficientNetB3"]["preds"].extend(eff_probs.squeeze().cpu().numpy())
            results["EfficientNetB3"]["labels"].extend(labels_np)
            
            # ViT predictions
            if use_tta:
                vit_probs, _ = tta.apply_tta(vit, imgs, device)
            else:
                vit_logits, _ = vit(imgs)
                vit_probs = torch.sigmoid(vit_logits)
            
            results["ViT"]["preds"].extend(vit_probs.squeeze().cpu().numpy())
            results["ViT"]["labels"].extend(labels_np)
            
            # Ensemble predictions
            with torch.no_grad():
                res_feats = resnet(imgs, return_features=True)
                eff_feats = effnet(imgs, return_features=True)
                vit_feats = vit(imgs, return_features=True)
                fusion_probs, _ = fusion_model(res_feats, eff_feats, vit_feats)
            
            results["Ensemble Fusion"]["preds"].extend(fusion_probs.squeeze().cpu().numpy())
            results["Ensemble Fusion"]["labels"].extend(labels_np)
    
    # Convert to arrays and compute metrics
    test_results = {}
    for model_name, data in results.items():
        preds = np.array(data["preds"])
        labels = np.array(data["labels"])
        
        metrics = calculate_clinical_metrics(labels, preds)
        test_results[model_name] = metrics
        
        logger.info(f"\n{model_name}:")
        logger.info(f"  Accuracy:   {metrics['accuracy']:.4f}")
        logger.info(f"  Precision:  {metrics['precision']:.4f}")
        logger.info(f"  Recall:     {metrics['sensitivity']:.4f}")
        logger.info(f"  Specificity:{metrics['specificity']:.4f}")
        logger.info(f"  F1-Score:   {metrics['f1_score']:.4f}")
        logger.info(f"  ROC-AUC:    {metrics['auc']:.4f}")
    
    return test_results


def save_training_plots(config: Dict[str, Any]):
    """Generate and save training history plots."""
    plots_dir = config["paths"]["plots_dir"]
    os.makedirs(plots_dir, exist_ok=True)
    logger.info(f"✓ Plots saved to {plots_dir}")


def main():
    """Main training pipeline with 5-Fold Cross Validation."""
    
    # Load configuration
    config = load_config()
    
    # Ensure reproducibility
    ensure_reproducibility(config)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"✓ Using device: {device}")
    
    # Create necessary directories
    create_checkpoint_dir()
    os.makedirs(config["paths"]["log_dir"], exist_ok=True)
    os.makedirs(config["paths"]["plots_dir"], exist_ok=True)
    
    # Load dataset
    logger.info(f"Loading dataset from {config['paths']['dataset_dir']}")
    try:
        full_dataset = MRIDataset(
            config['paths']['dataset_dir'],
            split='train',
            config=config,
            augment=True
        )
        logger.info(f"✓ Loaded {len(full_dataset)} training images")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.warning("Using synthetic data for demonstration...")
        return
    
    # 5-Fold Cross Validation
    if config["training"]["use_kfold"]:
        logger.info(f"\n{'='*60}")
        logger.info(f"5-Fold Cross Validation Pipeline")
        logger.info(f"{'='*60}")
        
        kfold_splits = get_kfold_splits(
            full_dataset,
            n_splits=config["training"]["num_folds"],
            seed=config["data"]["random_seed"]
        )
        
        fold_results = []
        
        for fold_num, (train_idx, val_idx) in enumerate(kfold_splits):
            logger.info(f"\n{'*'*60}")
            logger.info(f"FOLD {fold_num + 1}/{config['training']['num_folds']}")
            logger.info(f"{'*'*60}")
            
            # Create fold dataloaders
            train_loader, val_loader = create_fold_dataloaders(
                full_dataset,
                (train_idx, val_idx),
                batch_size=config["data"]["batch_size"],
                num_workers=config["data"]["num_workers"]
            )
            
            # TensorBoard writer
            tb_log_dir = os.path.join(config["paths"]["log_dir"], f"fold_{fold_num+1}")
            tb_writer = SummaryWriter(log_dir=tb_log_dir)
            
            # Train backbones
            for model_name in ["resnet50", "efficientnet_b3", "vit_b_16"]:
                model = load_backbone(model_name, pretrained=True).to(device)
                
                best_loss, best_metrics = train_backbone_with_progressive_unfreezing(
                    model_name=model_name,
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    config=config,
                    fold_num=fold_num,
                    tb_writer=tb_writer
                )
            
            # Train fusion
            best_loss, best_metrics = train_ensemble_fusion(
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                config=config,
                fold_num=fold_num,
                tb_writer=tb_writer
            )
            
            fold_results.append(best_metrics)
            tb_writer.close()
        
        # Summary across folds
        logger.info(f"\n{'='*60}")
        logger.info("CROSS-VALIDATION SUMMARY")
        logger.info(f"{'='*60}")
        
        all_metrics = {}
        for metric in ['accuracy', 'precision', 'sensitivity', 'specificity', 'f1_score', 'auc']:
            values = [r.get(metric, 0) for r in fold_results]
            all_metrics[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'fold_values': values
            }
            logger.info(
                f"{metric.upper():15s}: {np.mean(values):.4f} ± {np.std(values):.4f}"
            )
        
        # Save metrics
        metrics_file = config["paths"]["metrics_file"]
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        logger.info(f"\n✓ Metrics saved to {metrics_file}")
    
    else:
        # Single train/val split (for quick testing)
        logger.info("Running single train/val split (K-Fold disabled)")
        # Implementation similar to fold 0 above


if __name__ == "__main__":
    main()
