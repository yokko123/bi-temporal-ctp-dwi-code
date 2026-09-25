"""
Utility functions for mJNet training and testing.
"""

import os
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' or 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


class MetricTracker:
    """Track metrics during training."""
    
    def __init__(self):
        self.metrics = {}
    
    def update(self, name: str, value: float):
        """Update metric value."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def get(self, name: str) -> List[float]:
        """Get metric history."""
        return self.metrics.get(name, [])
    
    def get_last(self, name: str) -> Optional[float]:
        """Get last metric value."""
        values = self.get(name)
        return values[-1] if values else None
    
    def get_best(self, name: str, mode: str = "min") -> Optional[float]:
        """Get best metric value."""
        values = self.get(name)
        if not values:
            return None
        return min(values) if mode == "min" else max(values)
    
    def save(self, path: str):
        """Save metrics to file."""
        np.savez(path, **{k: np.array(v) for k, v in self.metrics.items()})
    
    def load(self, path: str):
        """Load metrics from file."""
        data = np.load(path)
        self.metrics = {k: list(data[k]) for k in data.files}


def plot_training_curves(
    metrics: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss curves
    ax = axes[0, 0]
    if "train_loss" in metrics:
        ax.plot(metrics["train_loss"], label="Train")
    if "val_loss" in metrics:
        ax.plot(metrics["val_loss"], label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True)
    
    # Dice curves
    ax = axes[0, 1]
    for name in ["dice_mean", "dice_penumbra", "dice_core"]:
        if name in metrics:
            ax.plot(metrics[name], label=name.replace("dice_", "").title())
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice Score")
    ax.set_title("Validation Dice Scores")
    ax.legend()
    ax.grid(True)
    
    # Learning rate
    ax = axes[1, 0]
    if "learning_rate" in metrics:
        ax.plot(metrics["learning_rate"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.set_yscale("log")
        ax.grid(True)
    
    # Individual class Dice
    ax = axes[1, 1]
    class_names = ["background", "brain", "penumbra", "core"]
    for name in class_names:
        key = f"dice_{name}"
        if key in metrics:
            ax.plot(metrics[key], label=name.title())
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice Score")
    ax.set_title("Per-Class Dice Scores")
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_prediction(
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    slice_idx: int,
    timepoint: int = 0,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Visualize a single slice with image, ground truth, and prediction.
    
    Args:
        image: Input image (T, Z, Y, X)
        ground_truth: Ground truth segmentation (Z, Y, X)
        prediction: Predicted segmentation (Z, Y, X)
        slice_idx: Z-slice index to visualize
        timepoint: Timepoint to show for the image
        save_path: Optional path to save the figure
        show: Whether to display the figure
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Color map for segmentation
    cmap = plt.cm.colors.ListedColormap(['black', 'gray', 'yellow', 'red'])
    
    # Input image
    ax = axes[0]
    ax.imshow(image[timepoint, slice_idx], cmap='gray')
    ax.set_title(f"Input (t={timepoint})")
    ax.axis('off')
    
    # Ground truth
    ax = axes[1]
    ax.imshow(ground_truth[slice_idx], cmap=cmap, vmin=0, vmax=3)
    ax.set_title("Ground Truth")
    ax.axis('off')
    
    # Prediction
    ax = axes[2]
    ax.imshow(prediction[slice_idx], cmap=cmap, vmin=0, vmax=3)
    ax.set_title("Prediction")
    ax.axis('off')
    
    # Overlay
    ax = axes[3]
    ax.imshow(image[timepoint, slice_idx], cmap='gray')
    mask = np.ma.masked_where(prediction[slice_idx] == 0, prediction[slice_idx])
    ax.imshow(mask, cmap=cmap, vmin=0, vmax=3, alpha=0.5)
    ax.set_title("Overlay")
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close()


def compute_volume_statistics(segmentation: np.ndarray, voxel_spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)) -> Dict[str, float]:
    """
    Compute volume statistics for segmentation.
    
    Args:
        segmentation: Segmentation mask (Z, Y, X)
        voxel_spacing: Voxel spacing in mm (z, y, x)
    
    Returns:
        Dictionary of volume statistics in ml
    """
    voxel_volume = np.prod(voxel_spacing) / 1000  # Convert mm³ to ml
    
    stats = {}
    class_names = ["background", "brain", "penumbra", "core"]
    
    for c, name in enumerate(class_names):
        voxel_count = np.sum(segmentation == c)
        stats[f"{name}_volume_ml"] = voxel_count * voxel_volume
    
    # Lesion = penumbra + core
    lesion_count = np.sum((segmentation == 2) | (segmentation == 3))
    stats["lesion_volume_ml"] = lesion_count * voxel_volume
    
    return stats


def normalize_volume(volume: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalize a 4D volume.
    
    Args:
        volume: Input volume (T, Z, Y, X)
        method: Normalization method ('minmax', 'zscore', 'percentile')
    
    Returns:
        Normalized volume
    """
    volume = volume.astype(np.float32)
    
    if method == "minmax":
        for t in range(volume.shape[0]):
            v_min, v_max = volume[t].min(), volume[t].max()
            if v_max > v_min:
                volume[t] = (volume[t] - v_min) / (v_max - v_min)
    
    elif method == "zscore":
        for t in range(volume.shape[0]):
            mean, std = volume[t].mean(), volume[t].std()
            if std > 0:
                volume[t] = (volume[t] - mean) / std
    
    elif method == "percentile":
        for t in range(volume.shape[0]):
            p1, p99 = np.percentile(volume[t], [1, 99])
            volume[t] = np.clip(volume[t], p1, p99)
            v_min, v_max = volume[t].min(), volume[t].max()
            if v_max > v_min:
                volume[t] = (volume[t] - v_min) / (v_max - v_min)
    
    return volume


def create_weight_map(
    label: np.ndarray,
    class_weights: List[float],
) -> np.ndarray:
    """
    Create a pixel-wise weight map based on class weights.
    
    Args:
        label: Label map (H, W) or (Z, Y, X)
        class_weights: Weight for each class
    
    Returns:
        Weight map with same shape as label
    """
    weight_map = np.zeros_like(label, dtype=np.float32)
    for c, w in enumerate(class_weights):
        weight_map[label == c] = w
    return weight_map
