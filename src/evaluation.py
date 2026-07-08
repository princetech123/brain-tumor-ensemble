import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import psutil
from typing import Dict, Any, List, Tuple

def calculate_clinical_metrics(y_true: np.ndarray, y_scores: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Calculates clinical diagnostic metrics from ground truths and tumor probabilities.
    
    Metrics: Accuracy, Precision, Recall/Sensitivity, Specificity, F1, AUC.
    """
    y_pred = (y_scores >= threshold).astype(int)
    
    # Calculate confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    accuracy = float((tp + tn) / (tp + tn + fp + fn))
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0  # Same as recall
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(2 * (precision * sensitivity) / (precision + sensitivity)) if (precision + sensitivity) > 0 else 0.0
    
    # Calculate AUC
    try:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = float(auc(fpr, tpr))
    except Exception:
        roc_auc = 0.0
        
    return {
        "accuracy": accuracy,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1_score": f1,
        "auc": roc_auc,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn)
    }

def plot_confusion_matrix(tn: int, fp: int, fn: int, tp: int, save_path: str):
    """Generates and saves a clean, dark-themed Confusion Matrix plot."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    cm = np.array([[tn, fp], [fn, tp]])
    labels = ["Normal", "Tumor"]
    
    plt.figure(figsize=(6, 5), facecolor='#0d1117')
    ax = plt.subplot()
    ax.set_facecolor('#0d1117')
    
    # Custom colored matrix heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im)
    
    # Text tags
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14)
            
    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(labels, color='white', fontsize=12)
    ax.set_yticklabels(labels, color='white', fontsize=12)
    
    ax.set_xlabel('Predicted Label', color='white', fontsize=12)
    ax.set_ylabel('True Label', color='white', fontsize=12)
    ax.set_title('Confusion Matrix', color='white', fontsize=14, pad=15)
    
    # Adjust border colors
    for spine in ax.spines.values():
        spine.set_color('#30363d')
        
    plt.tight_layout()
    plt.savefig(save_path, facecolor='#0d1117', dpi=150)
    plt.close()

def plot_roc_curves(model_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]], save_path: str):
    """Plots comparative ROC curves for all models and saves the plot.
    
    Args:
        model_predictions: Dictionary mapping model_name -> (y_true, y_scores)
        save_path: Location to write the PNG image.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(7, 6), facecolor='#0d1117')
    ax = plt.subplot()
    ax.set_facecolor('#0d1117')
    
    colors = {
        "ResNet50": "#ff5a79",
        "EfficientNetB3": "#20b2aa",
        "ViT": "#ffaa00",
        "Ensemble Fusion": "#00f0ff"
    }
    
    for model_name, (y_true, y_scores) in model_predictions.items():
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        color = colors.get(model_name, "#ffffff")
        plt.plot(fpr, tpr, color=color, lw=2.5, label=f'{model_name} (AUC = {roc_auc:.3f})')
        
    plt.plot([0, 1], [0, 1], color='#30363d', lw=1.5, linestyle='--')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    
    plt.xlabel('False Positive Rate (1 - Specificity)', color='white', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', color='white', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves', color='white', fontsize=14, pad=15)
    
    ax.tick_params(colors='white')
    ax.set_xticks(np.arange(0.0, 1.1, 0.1))
    ax.set_yticks(np.arange(0.0, 1.1, 0.1))
    
    # Style the legend
    legend = plt.legend(loc="lower right", facecolor='#161b22', edgecolor='#30363d')
    for text in legend.get_texts():
        text.set_color('white')
        
    for spine in ax.spines.values():
        spine.set_color('#30363d')
        
    plt.grid(True, color='#21262d', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, facecolor='#0d1117', dpi=150)
    plt.close()

def get_system_metrics() -> Dict[str, float]:
    """Retrieves current memory usage and process info."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        "memory_rss_mb": float(memory_info.rss / (1024 * 1024)), # MB
        "cpu_percent": float(psutil.cpu_percent())
    }
