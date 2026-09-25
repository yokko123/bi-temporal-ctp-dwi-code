"""
Loss functions for mJNet training.
Includes Dice, Focal, Tversky, and combined losses.

NOTE: The squared_dice_coef_loss matches the reference TensorFlow implementation
      from CNN-segmentation-of-infarcted-regions exactly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class SquaredDiceLoss(nn.Module):
    """
    Squared Dice Loss - EXACT match to the reference TensorFlow implementation.
    
    Reference: CNN-segmentation-of-infarcted-regions/Utils/metrics.py::_squared_dice_coef
    
    The weights are applied to BOTH numerator and denominator (not as a weighted mean),
    which is mathematically different from standard weighted Dice.
    
    Formula:
        numerator = 2 * sum(y_true * y_pred * weights)
        denominator = sum((y_true^2 + y_pred^2) * weights)
        dice = numerator / denominator
        loss = 1 - dice
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        class_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        
        # Default weights: [0.0, 0.1, 50, 440] for [bg, brain, penumbra, core]
        # Background weight = 0.0: excluded from loss (paper excludes background).
        if class_weights is None:
            class_weights = [0.0, 0.1, 50.0, 440.0]
        self.register_buffer('class_weights', torch.tensor(class_weights).float())
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions (B, C, H, W) - raw logits
            target: Ground truth (B, H, W) - class indices
        
        Returns:
            Squared Dice loss value (scalar)
        """
        # Apply softmax to get probabilities
        pred = F.softmax(pred, dim=1)  # (B, C, H, W)
        
        # Convert target to one-hot
        target_one_hot = F.one_hot(target, self.num_classes)  # (B, H, W, C)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)
        
        # Reshape weights for broadcasting: (1, C, 1, 1)
        weights = self.class_weights.view(1, -1, 1, 1)
        
        # Weighted numerator: 2 * sum(y_true * y_pred * weights)
        numerator = 2.0 * (target_one_hot * pred * weights).sum(dim=(1, 2, 3))  # (B,)
        
        # Weighted denominator: sum((y_true^2 + y_pred^2) * weights)
        denominator = ((target_one_hot ** 2 + pred ** 2) * weights).sum(dim=(1, 2, 3))  # (B,)
        
        # Dice coefficient per sample
        dice = numerator / (denominator + 1e-6)  # (B,)
        
        # Loss = 1 - mean(dice)
        return 1.0 - dice.mean()


class DiceLoss(nn.Module):
    """
    Standard Dice Loss for multi-class segmentation.
    (Alternative to SquaredDiceLoss with different weighting semantics)
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        smooth: float = 1e-6,
        class_weights: Optional[List[float]] = None,
        squared: bool = True,
    ):
        """
        Args:
            num_classes: Number of classes
            smooth: Smoothing factor to avoid division by zero
            class_weights: Optional weights for each class
            squared: If True, use squared Dice coefficient
        """
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.squared = squared
        
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights).float())
        else:
            self.class_weights = None
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions of shape (B, C, H, W) - raw logits
            target: Ground truth of shape (B, H, W) - class indices
        
        Returns:
            Dice loss value
        """
        # Apply softmax to predictions
        pred = F.softmax(pred, dim=1)
        
        # Convert target to one-hot encoding
        target_one_hot = F.one_hot(target, self.num_classes)  # (B, H, W, C)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)
        
        # Calculate Dice for each class
        if self.squared:
            intersection = (pred * target_one_hot).sum(dim=(2, 3))
            union = (pred ** 2).sum(dim=(2, 3)) + (target_one_hot ** 2).sum(dim=(2, 3))
        else:
            intersection = (pred * target_one_hot).sum(dim=(2, 3))
            union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Apply class weights via weighted mean (not direct multiplication)
        if self.class_weights is not None:
            # Weighted mean: sum(w_i * dice_i) / sum(w_i)
            weights = self.class_weights.unsqueeze(0).expand_as(dice)  # (B, C)
            weighted_dice = (dice * weights).sum(dim=1) / weights.sum(dim=1)
            return 1.0 - weighted_dice.mean()
        
        # Return mean Dice loss (unweighted)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        alpha: Optional[List[float]] = None,
        gamma: float = 2.0,
    ):
        """
        Args:
            num_classes: Number of classes
            alpha: Class weights (balancing factor)
            gamma: Focusing parameter
        """
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        
        if alpha is not None:
            self.register_buffer('alpha', torch.tensor(alpha).float())
        else:
            self.alpha = None
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions of shape (B, C, H, W) - raw logits
            target: Ground truth of shape (B, H, W) - class indices
        
        Returns:
            Focal loss value
        """
        # Apply softmax
        pred_softmax = F.softmax(pred, dim=1)
        
        # Get probabilities for target classes
        target_one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        pt = (pred_softmax * target_one_hot).sum(dim=1)  # (B, H, W)
        
        # Compute focal loss
        focal_weight = (1 - pt) ** self.gamma
        ce_loss = F.cross_entropy(pred, target, reduction='none')  # (B, H, W)
        
        focal_loss = focal_weight * ce_loss
        
        # Apply class-specific alpha weights
        if self.alpha is not None:
            alpha_t = self.alpha[target]  # (B, H, W)
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()


class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice loss.
    Allows controlling the penalty for false positives vs false negatives.
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1e-6,
        class_weights: Optional[List[float]] = None,
    ):
        """
        Args:
            num_classes: Number of classes
            alpha: Weight for false positives
            beta: Weight for false negatives
            smooth: Smoothing factor
            class_weights: Optional class weights
        """
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights).float())
        else:
            self.class_weights = None
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions of shape (B, C, H, W) - raw logits
            target: Ground truth of shape (B, H, W) - class indices
        
        Returns:
            Tversky loss value
        """
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        
        # True positives, false positives, false negatives
        tp = (pred * target_one_hot).sum(dim=(2, 3))
        fp = (pred * (1 - target_one_hot)).sum(dim=(2, 3))
        fn = ((1 - pred) * target_one_hot).sum(dim=(2, 3))
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        if self.class_weights is not None:
            tversky = tversky * self.class_weights.unsqueeze(0)
        
        return 1.0 - tversky.mean()


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss - combines Tversky with focal mechanism.
    
    beta > alpha penalizes false negatives more than false positives,
    which helps detect rare classes like core.
    
    Background is excluded when class_weights[0] = 0.
    
    NOTE: Unlike SquaredDiceLoss, this loss does NOT use magnitude-based
    class weights. The alpha/beta/gamma parameters already handle class
    imbalance. class_weights are used only as a BINARY MASK to decide
    which classes participate in the loss (0 = excluded, >0 = included).
    All active classes are weighted equally.

    Label smoothing (default 0.05) converts hard one-hot targets to soft
    targets, preventing overconfident predictions and reducing overfitting.
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        alpha: float = 0.2,
        beta: float = 0.8,
        gamma: float = 1.5,
        smooth: float = 1e-6,
        class_weights: Optional[List[float]] = None,
        label_smoothing: float = 0.02,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.label_smoothing = label_smoothing
        
        # Build binary active mask from class_weights
        # Any weight > 0 means the class is active; the magnitude is ignored.
        if class_weights is not None:
            active = torch.tensor([1.0 if w > 0 else 0.0 for w in class_weights])
        else:
            # Default: exclude background, include all foreground
            active = torch.tensor([0.0, 1.0, 1.0, 1.0])
        self.register_buffer('active_mask', active)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()

        # Label smoothing: hard [0,1] -> soft [eps/C, 1-eps+eps/C]
        if self.label_smoothing > 0:
            eps = self.label_smoothing
            target_one_hot = target_one_hot * (1.0 - eps) + eps / self.num_classes
        
        tp = (pred * target_one_hot).sum(dim=(2, 3))
        fp = (pred * (1 - target_one_hot)).sum(dim=(2, 3))
        fn = ((1 - pred) * target_one_hot).sum(dim=(2, 3))
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal_tversky = (1 - tversky) ** self.gamma  # (B, C)
        
        # Equal weight for all active classes; exclude inactive ones (background)
        masked = focal_tversky * self.active_mask.unsqueeze(0)  # (B, C)
        n_active = self.active_mask.sum().clamp(min=1.0)
        return masked.sum(dim=1).mean() / n_active


class CombinedLoss(nn.Module):
    """
    Combined loss function (Dice + Cross Entropy).
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        class_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        
        self.dice_loss = DiceLoss(num_classes=num_classes, class_weights=class_weights)
        
        if class_weights is not None:
            self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).float())
        else:
            self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(pred, target)
        ce = self.ce_loss(pred, target)
        return self.dice_weight * dice + self.ce_weight * ce


def get_loss_function(
    loss_name: str,
    num_classes: int = 4,
    class_weights: Optional[List[float]] = None,
) -> nn.Module:
    """
    Factory function to get loss function by name.
    
    Args:
        loss_name: Name of the loss function
        num_classes: Number of classes
        class_weights: Optional class weights
    
    Returns:
        Loss function module
    """
    loss_name = loss_name.lower()
    
    if loss_name == "dice" or loss_name == "squared_dice":
        # Use SquaredDiceLoss matching the reference TF implementation
        # Default weights: [0.0, 0.1, 50, 440] — background excluded from loss
        return SquaredDiceLoss(num_classes=num_classes, class_weights=class_weights)
    elif loss_name == "dice_standard":
        # Alternative: standard per-class dice with weighted mean
        return DiceLoss(num_classes=num_classes, class_weights=class_weights)
    elif loss_name == "focal":
        return FocalLoss(num_classes=num_classes, alpha=class_weights)
    elif loss_name == "tversky":
        return TverskyLoss(num_classes=num_classes, class_weights=class_weights)
    elif loss_name == "focal_tversky":
        return FocalTverskyLoss(num_classes=num_classes, class_weights=class_weights)
    elif loss_name == "ce" or loss_name == "cross_entropy":
        if class_weights is not None:
            return nn.CrossEntropyLoss(weight=torch.tensor(class_weights).float())
        return nn.CrossEntropyLoss()
    elif loss_name == "combined":
        return CombinedLoss(num_classes=num_classes, class_weights=class_weights)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


# Metrics for evaluation
class DiceScore(nn.Module):
    """Compute Dice score for each class."""
    
    def __init__(self, num_classes: int = 4, smooth: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions of shape (B, C, H, W) - raw logits or probabilities
            target: Ground truth of shape (B, H, W) - class indices
        
        Returns:
            Dice scores for each class, shape (C,)
        """
        if pred.shape[1] == self.num_classes:
            pred = F.softmax(pred, dim=1)
        
        target_one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        
        intersection = (pred * target_one_hot).sum(dim=(0, 2, 3))
        union = pred.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return dice


class IoUScore(nn.Module):
    """Compute IoU (Jaccard) score for each class."""
    
    def __init__(self, num_classes: int = 4, smooth: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape[1] == self.num_classes:
            pred = F.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1)
        
        ious = []
        for c in range(self.num_classes):
            pred_c = (pred == c).float()
            target_c = (target == c).float()
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum() - intersection
            
            iou = (intersection + self.smooth) / (union + self.smooth)
            ious.append(iou)
        
        return torch.stack(ious)
