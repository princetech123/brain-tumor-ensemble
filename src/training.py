import os
import sys

# Ensure the project root is in the path when running this script directly
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import yaml
import time
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from src.preprocessing import MRIDataset
from src.models.backbones import load_backbone
from src.models.fusion import EnsembleFusionModel
from src.evaluation import calculate_clinical_metrics, plot_confusion_matrix, plot_roc_curves, get_system_metrics

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience: int = 3, delta: float = 0.0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss >= self.best_loss - self.delta:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop

def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def train_backbone(model_name: str, config: Dict[str, Any], train_loader: DataLoader, val_loader: DataLoader, device: torch.device) -> str:
    """Fine-tunes a single backbone model (Stage 1)."""
    logger.info(f"--- Training Backbone: {model_name} ---")
    
    # Load model
    model = load_backbone(model_name, pretrained=False).to(device)  # Set to True if internet is available to download pretrained weights
    
    # Paths
    checkpoint_dir = config["paths"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}.pth")
    
    # Tensorboard logger
    tb_writer = SummaryWriter(log_dir=os.path.join(config["paths"]["log_dir"], model_name))
    
    # Hyperparameters
    train_cfg = config["training"]["backbone"]
    epochs = train_cfg["epochs"]
    initial_lr = train_cfg["learning_rate"]
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=train_cfg["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=train_cfg["lr_reduction_factor"], 
        patience=train_cfg["lr_reduction_patience"]
    )
    early_stopping = EarlyStopping(patience=train_cfg["early_stopping_patience"])
    
    # Phase 1: Freeze backbone, train classification head first (2 epochs)
    logger.info("Phase 1: Training standalone classifier head...")
    model.freeze_backbone(True)
    for epoch in range(min(2, epochs)):
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            logits, _ = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            
    # Phase 2: Unfreeze last blocks and fine-tune (remaining epochs)
    logger.info("Phase 2: Fine-tuning last layers...")
    model.unfreeze_last_blocks()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            logits, _ = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation pass
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
                logits, _ = model(imgs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * imgs.size(0)
                
                probs = torch.sigmoid(logits)
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        metrics = calculate_clinical_metrics(all_labels, all_preds)
        
        # Logging
        tb_writer.add_scalar("Loss/Train", train_loss, epoch)
        tb_writer.add_scalar("Loss/Val", val_loss, epoch)
        tb_writer.add_scalar("Accuracy/Val", metrics["accuracy"], epoch)
        tb_writer.add_scalar("F1_Score/Val", metrics["f1_score"], epoch)
        
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {metrics['accuracy']:.4f} | Val F1: {metrics['f1_score']:.4f}")
        
        # Checkpoint saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved best model checkpoint to {checkpoint_path}")
            
        # Early stopping
        if early_stopping(val_loss):
            logger.info("Early stopping triggered. Ending training.")
            break
            
    tb_writer.close()
    return checkpoint_path

def train_ensemble_fusion(config: Dict[str, Any], train_loader: DataLoader, val_loader: DataLoader, device: torch.device):
    """Trains the Attention Fusion layer (Stage 2) using features from fine-tuned backbones."""
    logger.info("--- Training Ensemble Fusion Model ---")
    
    # 1. Load fine-tuned backbones and freeze them
    checkpoint_dir = config["paths"]["checkpoint_dir"]
    
    resnet = load_backbone("resnet50", pretrained=False).to(device)
    resnet.load_state_dict(torch.load(os.path.join(checkpoint_dir, "resnet50.pth"), map_location=device))
    resnet.eval()
    
    effnet = load_backbone("efficientnet_b3", pretrained=False).to(device)
    effnet.load_state_dict(torch.load(os.path.join(checkpoint_dir, "efficientnet_b3.pth"), map_location=device))
    effnet.eval()
    
    vit = load_backbone("vit_b_16", pretrained=False).to(device)
    vit.load_state_dict(torch.load(os.path.join(checkpoint_dir, "vit_b_16.pth"), map_location=device))
    vit.eval()
    
    # 2. Instantiate Fusion model
    fusion_model = EnsembleFusionModel(config).to(device)
    
    # Hyperparameters
    train_cfg = config["training"]["fusion"]
    epochs = train_cfg["epochs"]
    lr = train_cfg["learning_rate"]
    
    # Loss and optimizer (Only update fusion_model parameters!)
    criterion = nn.BCELoss()  # Output of Fusion model has Sigmoid already
    optimizer = optim.AdamW(fusion_model.parameters(), lr=lr, weight_decay=train_cfg["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=train_cfg["lr_reduction_factor"], 
        patience=train_cfg["lr_reduction_patience"]
    )
    early_stopping = EarlyStopping(patience=train_cfg["early_stopping_patience"])
    
    tb_writer = SummaryWriter(log_dir=os.path.join(config["paths"]["log_dir"], "fusion"))
    best_val_loss = float('inf')
    checkpoint_path = os.path.join(checkpoint_dir, "fusion.pth")
    
    for epoch in range(epochs):
        fusion_model.train()
        train_loss = 0.0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
            
            # Extract features from frozen backbones
            with torch.no_grad():
                res_feats = resnet(imgs, return_features=True)
                eff_feats = effnet(imgs, return_features=True)
                vit_feats = vit(imgs, return_features=True)
                
            optimizer.zero_grad()
            # Forward pass on fusion network
            probs, _ = fusion_model(res_feats, eff_feats, vit_feats)
            loss = criterion(probs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation pass
        fusion_model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
                
                res_feats = resnet(imgs, return_features=True)
                eff_feats = effnet(imgs, return_features=True)
                vit_feats = vit(imgs, return_features=True)
                
                probs, _ = fusion_model(res_feats, eff_feats, vit_feats)
                loss = criterion(probs, labels)
                val_loss += loss.item() * imgs.size(0)
                
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        metrics = calculate_clinical_metrics(all_labels, all_preds)
        
        tb_writer.add_scalar("Loss/Train", train_loss, epoch)
        tb_writer.add_scalar("Loss/Val", val_loss, epoch)
        tb_writer.add_scalar("Accuracy/Val", metrics["accuracy"], epoch)
        tb_writer.add_scalar("F1_Score/Val", metrics["f1_score"], epoch)
        
        logger.info(f"Fusion Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {metrics['accuracy']:.4f} | Val F1: {metrics['f1_score']:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(fusion_model.state_dict(), checkpoint_path)
            logger.info(f"Saved best fusion model checkpoint to {checkpoint_path}")
            
        if early_stopping(val_loss):
            logger.info("Early stopping triggered. Ending fusion training.")
            break
            
    tb_writer.close()
    return checkpoint_path

def evaluate_all_on_test(config: Dict[str, Any], test_loader: DataLoader, device: torch.device):
    """Evaluates all standalone models and the ensemble model on the test dataset."""
    logger.info("--- Starting Comprehensive Model Evaluation on Test Set ---")
    
    checkpoint_dir = config["paths"]["checkpoint_dir"]
    
    # Load all models
    resnet = load_backbone("resnet50", pretrained=False).to(device)
    resnet.load_state_dict(torch.load(os.path.join(checkpoint_dir, "resnet50.pth"), map_location=device))
    resnet.eval()
    
    effnet = load_backbone("efficientnet_b3", pretrained=False).to(device)
    effnet.load_state_dict(torch.load(os.path.join(checkpoint_dir, "efficientnet_b3.pth"), map_location=device))
    effnet.eval()
    
    vit = load_backbone("vit_b_16", pretrained=False).to(device)
    vit.load_state_dict(torch.load(os.path.join(checkpoint_dir, "vit_b_16.pth"), map_location=device))
    vit.eval()
    
    fusion_model = EnsembleFusionModel(config).to(device)
    fusion_model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "fusion.pth"), map_location=device))
    fusion_model.eval()
    
    predictions = {
        "ResNet50": ([], []),
        "EfficientNetB3": ([], []),
        "ViT": ([], []),
        "Ensemble Fusion": ([], [])
    }
    
    # To measure inference times
    inf_times = {k: [] for k in predictions.keys()}
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels_np = labels.numpy()
            
            # ResNet50
            start = time.perf_counter()
            res_logits, res_feats = resnet(imgs)
            res_probs = torch.sigmoid(res_logits).cpu().numpy().squeeze(1)
            inf_times["ResNet50"].append((time.perf_counter() - start) / imgs.size(0))
            predictions["ResNet50"][0].extend(labels_np)
            predictions["ResNet50"][1].extend(res_probs)
            
            # EfficientNetB3
            start = time.perf_counter()
            eff_logits, eff_feats = effnet(imgs)
            eff_probs = torch.sigmoid(eff_logits).cpu().numpy().squeeze(1)
            inf_times["EfficientNetB3"].append((time.perf_counter() - start) / imgs.size(0))
            predictions["EfficientNetB3"][0].extend(labels_np)
            predictions["EfficientNetB3"][1].extend(eff_probs)
            
            # ViT
            start = time.perf_counter()
            vit_logits, vit_feats = vit(imgs)
            vit_probs = torch.sigmoid(vit_logits).cpu().numpy().squeeze(1)
            inf_times["ViT"].append((time.perf_counter() - start) / imgs.size(0))
            predictions["ViT"][0].extend(labels_np)
            predictions["ViT"][1].extend(vit_probs)
            
            # Ensemble Fusion
            start = time.perf_counter()
            fusion_probs, _ = fusion_model(res_feats, eff_feats, vit_feats)
            fusion_probs_np = fusion_probs.cpu().numpy().squeeze(1)
            # Add total chain execution time
            inf_times["Ensemble Fusion"].append((time.perf_counter() - start) / imgs.size(0))
            predictions["Ensemble Fusion"][0].extend(labels_np)
            predictions["Ensemble Fusion"][1].extend(fusion_probs_np)
            
    # Calculate performance metrics
    metrics_report = {}
    model_predictions_for_roc = {}
    
    # Directory to store static assets for web front-end
    static_eval_dir = "src/web/static/eval"
    os.makedirs(static_eval_dir, exist_ok=True)
    
    for model_name, (y_true, y_scores) in predictions.items():
        y_true_np = np.array(y_true)
        y_scores_np = np.array(y_scores)
        
        metrics = calculate_clinical_metrics(y_true_np, y_scores_np)
        
        # Add latency and memory info
        metrics["inference_latency_ms"] = float(np.mean(inf_times[model_name]) * 1000)
        sys_metrics = get_system_metrics()
        metrics["memory_usage_mb"] = sys_metrics["memory_rss_mb"]
        
        metrics_report[model_name] = metrics
        model_predictions_for_roc[model_name] = (y_true_np, y_scores_np)
        
        # Generate Confusion Matrix for individual model
        if model_name == "Ensemble Fusion":
            plot_confusion_matrix(
                metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"],
                os.path.join(static_eval_dir, "confusion_matrix.png")
            )
            
    # Generate ROC Curve plots
    plot_roc_curves(model_predictions_for_roc, os.path.join(static_eval_dir, "roc_curve.png"))
    
    # Save statistics report
    metrics_path = config["paths"]["metrics_file"]
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics_report, f, indent=4)
        
    logger.info(f"Diagnostic metrics successfully written to '{metrics_path}'")
    logger.info("Evaluation Complete.")

def main():
    config = load_config()
    
    # Hardware device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using execution device: {device}")
    
    dataset_dir = config["paths"]["dataset_dir"]
    batch_size = config["data"]["batch_size"]
    
    # Datasets and loaders
    train_dataset = MRIDataset(dataset_dir, "train", config, augment=True)
    val_dataset = MRIDataset(dataset_dir, "val", config, augment=False)
    test_dataset = MRIDataset(dataset_dir, "test", config, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=config["data"]["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=config["data"]["num_workers"])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=config["data"]["num_workers"])
    
    # Stage 1: Fine-tune individual backbones
    train_backbone("resnet50", config, train_loader, val_loader, device)
    train_backbone("efficientnet_b3", config, train_loader, val_loader, device)
    train_backbone("vit_b_16", config, train_loader, val_loader, device)
    
    # Stage 2: Train Feature Fusion Ensemble
    train_ensemble_fusion(config, train_loader, val_loader, device)
    
    # Evaluation
    evaluate_all_on_test(config, test_loader, device)

if __name__ == "__main__":
    main()
