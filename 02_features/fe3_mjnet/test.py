"""
Testing/inference script for mJNet on 4D CTP data.
Performs inference on full volumes and saves predictions as NIfTI files.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from Data.dataset import get_test_loader
from Utils.losses import DiceScore, IoUScore
from Architectures.arch_mJNet import create_mjnet


def setup_logging(log_dir: str) -> logging.Logger:
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"test_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_model(
    checkpoint_path: str,
    num_classes: int,
    timepoints: int,
    patch_size: int,
    longJ: bool,
    v2: bool,
    device: torch.device,
) -> nn.Module:
    """Load model from checkpoint."""
    model = create_mjnet(
        input_shape=(timepoints, patch_size, patch_size),
        num_classes=num_classes,
        longJ=longJ,
        v2=v2,
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    return model


def predict_volume(
    model: nn.Module,
    volume: np.ndarray,
    device: torch.device,
    patch_size: int = 16,
    stride: int = 4,
    batch_size: int = 64,
    clip_min: float = 0.0,
    clip_max: float = 255.0,
) -> np.ndarray:
    """
    Predict segmentation for a full 4D volume using patch-based inference.
    
    Args:
        model: Trained model
        volume: Input volume of shape (T, Z, Y, X)
        device: Torch device
        patch_size: Size of patches
        stride: Stride for patch extraction
        batch_size: Batch size for inference
        clip_min: Normalisation lower bound (must match training)
        clip_max: Normalisation upper bound (must match training)
    
    Returns:
        Prediction volume of shape (Z, Y, X)
    """
    T, Z, Y, X = volume.shape
    
    # Initialize prediction and count arrays
    prediction = np.zeros((4, Z, Y, X), dtype=np.float32)  # (C, Z, Y, X)
    count = np.zeros((Z, Y, X), dtype=np.float32)
    
    # Normalize volume — global clip normalisation (same as training)
    volume_norm = (volume.astype(np.float32) - clip_min) / (clip_max - clip_min + 1e-8)
    volume_norm = np.clip(volume_norm, 0.0, 1.0)
    
    # Extract patches for each z-slice
    with torch.no_grad():
        for z in tqdm(range(Z), desc="Processing slices", leave=False):
            patches = []
            positions = []
            
            # Extract patches from this slice
            for y in range(0, Y - patch_size + 1, stride):
                for x in range(0, X - patch_size + 1, stride):
                    patch = volume_norm[:, z, y:y+patch_size, x:x+patch_size]  # (T, H, W)
                    patches.append(patch)
                    positions.append((y, x))
            
            if not patches:
                continue
            
            # Process in batches
            patches = np.array(patches)  # (N, T, H, W)
            
            for i in range(0, len(patches), batch_size):
                batch_patches = torch.from_numpy(patches[i:i+batch_size]).float().to(device)
                batch_patches = batch_patches.unsqueeze(1)  # (B, 1, T, H, W)
                
                outputs = model(batch_patches)  # (B, C, H, W)
                probs = F.softmax(outputs, dim=1)  # (B, C, H, W)
                probs = probs.cpu().numpy()
                
                # Accumulate predictions
                for j, (y, x) in enumerate(positions[i:i+batch_size]):
                    prediction[:, z, y:y+patch_size, x:x+patch_size] += probs[j]
                    count[z, y:y+patch_size, x:x+patch_size] += 1
    
    # Average overlapping predictions
    count = np.maximum(count, 1e-6)
    prediction = prediction / count[np.newaxis, :, :, :]
    
    # Get final segmentation
    segmentation = np.argmax(prediction, axis=0)  # (Z, Y, X)
    
    return segmentation.astype(np.int16)


def predict_volume_sliding_window(
    model: nn.Module,
    volume: np.ndarray,
    device: torch.device,
    patch_size: int = 16,
    overlap: float = 0.5,
) -> np.ndarray:
    """
    Alternative sliding window inference with configurable overlap.
    """
    T, Z, Y, X = volume.shape
    stride = int(patch_size * (1 - overlap))
    
    return predict_volume(
        model=model,
        volume=volume,
        device=device,
        patch_size=patch_size,
        stride=stride,
    )


def evaluate_prediction(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    num_classes: int = 4,
) -> Dict[str, float]:
    """
    Evaluate prediction against ground truth.
    
    Args:
        prediction: Predicted segmentation (Z, Y, X)
        ground_truth: Ground truth segmentation (Z, Y, X)
        num_classes: Number of classes
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Class names
    class_names = ["background", "brain", "penumbra", "core"]
    
    # Compute Dice and IoU for each class
    for c in range(num_classes):
        pred_c = (prediction == c).astype(np.float32)
        gt_c = (ground_truth == c).astype(np.float32)
        
        intersection = (pred_c * gt_c).sum()
        union = pred_c.sum() + gt_c.sum()
        
        # Dice
        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        metrics[f"dice_{class_names[c]}"] = dice
        
        # IoU
        iou = (intersection + 1e-6) / (union - intersection + 1e-6)
        metrics[f"iou_{class_names[c]}"] = iou
    
    # Mean metrics (excluding background)
    metrics["dice_mean"] = np.mean([metrics[f"dice_{class_names[c]}"] for c in range(1, num_classes)])
    metrics["iou_mean"] = np.mean([metrics[f"iou_{class_names[c]}"] for c in range(1, num_classes)])
    
    # Lesion-specific metrics (penumbra + core)
    pred_lesion = ((prediction == 2) | (prediction == 3)).astype(np.float32)
    gt_lesion = ((ground_truth == 2) | (ground_truth == 3)).astype(np.float32)
    
    intersection = (pred_lesion * gt_lesion).sum()
    union = pred_lesion.sum() + gt_lesion.sum()
    
    metrics["dice_lesion"] = (2 * intersection + 1e-6) / (union + 1e-6)
    
    return metrics


def save_prediction(
    prediction: np.ndarray,
    reference_nifti: nib.Nifti1Image,
    output_path: str,
):
    """Save prediction as NIfTI file with same header as reference."""
    # Create NIfTI image with same affine and header
    pred_nifti = nib.Nifti1Image(prediction, reference_nifti.affine, reference_nifti.header)
    nib.save(pred_nifti, output_path)


def test(config: Config, args: argparse.Namespace):
    """Main testing function."""
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logging(args.output_dir)
    
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from: {args.checkpoint}")
    model = load_model(
        checkpoint_path=args.checkpoint,
        num_classes=config.data.num_classes,
        timepoints=config.data.timepoints,
        patch_size=config.data.patch_size,
        longJ=args.longJ,
        v2=args.v2,
        device=device,
    )
    
    # Get test files
    data_dir = Path(args.data_dir)
    label_dir = Path(args.label_dir) if args.label_dir else None
    
    # Find all NIfTI files
    nifti_files = sorted(list(data_dir.glob("*.nii")) + list(data_dir.glob("*.nii.gz")))
    logger.info(f"Found {len(nifti_files)} test volumes")
    
    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each volume
    all_metrics = []
    
    for nifti_path in tqdm(nifti_files, desc="Testing"):
        logger.info(f"\nProcessing: {nifti_path.name}")
        
        # Load volume
        nifti = nib.load(str(nifti_path))
        volume = nifti.get_fdata()
        
        # Handle different dimension orders
        if volume.ndim == 4:
            if volume.shape[-1] < volume.shape[0]:
                # Assume (X, Y, Z, T) format - transpose to (T, Z, Y, X)
                volume = np.transpose(volume, (3, 2, 1, 0))
            else:
                # Assume (T, Z, Y, X) format
                pass
        else:
            logger.warning(f"Unexpected volume shape: {volume.shape}, skipping")
            continue
        
        logger.info(f"Volume shape (T, Z, Y, X): {volume.shape}")
        
        # Predict
        prediction = predict_volume(
            model=model,
            volume=volume,
            device=device,
            patch_size=config.data.patch_size,
            stride=args.stride,
            batch_size=args.batch_size,
            clip_min=config.data.clip_min,
            clip_max=config.data.clip_max,
        )
        
        # Save prediction
        output_path = os.path.join(args.output_dir, f"{nifti_path.stem}_pred.nii.gz")
        
        # Transpose prediction back if needed (Z, Y, X) -> original orientation
        pred_save = np.transpose(prediction, (2, 1, 0))  # (X, Y, Z)
        save_prediction(pred_save, nifti, output_path)
        logger.info(f"Saved prediction to: {output_path}")
        
        # Evaluate if labels available
        if label_dir is not None:
            label_path = label_dir / nifti_path.name
            if not label_path.exists():
                # Try with different extension
                label_path = label_dir / f"{nifti_path.stem.replace('.nii', '')}.nii.gz"
            
            if label_path.exists():
                label_nifti = nib.load(str(label_path))
                ground_truth = label_nifti.get_fdata()
                
                # Handle dimension order
                if ground_truth.ndim == 3:
                    ground_truth = np.transpose(ground_truth, (2, 1, 0))  # (X, Y, Z) -> (Z, Y, X)
                
                metrics = evaluate_prediction(prediction, ground_truth.astype(np.int16))
                all_metrics.append(metrics)
                
                logger.info(f"Dice - Brain: {metrics['dice_brain']:.4f}, "
                           f"Penumbra: {metrics['dice_penumbra']:.4f}, "
                           f"Core: {metrics['dice_core']:.4f}")
                logger.info(f"Mean Dice: {metrics['dice_mean']:.4f}")
                logger.info(f"Lesion Dice: {metrics['dice_lesion']:.4f}")
    
    # Summary
    if all_metrics:
        logger.info("\n" + "=" * 50)
        logger.info("Overall Results:")
        
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            mean_val = np.mean(values)
            std_val = np.std(values)
            logger.info(f"{key}: {mean_val:.4f} ± {std_val:.4f}")
        
        # Save metrics to file
        metrics_path = os.path.join(args.output_dir, "metrics.txt")
        with open(metrics_path, "w") as f:
            f.write("Subject-wise Metrics:\n")
            for i, (nifti_path, metrics) in enumerate(zip(nifti_files, all_metrics)):
                f.write(f"\n{nifti_path.name}:\n")
                for key, value in metrics.items():
                    f.write(f"  {key}: {value:.4f}\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("Overall Metrics:\n")
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics]
                f.write(f"  {key}: {np.mean(values):.4f} ± {np.std(values):.4f}\n")
        
        logger.info(f"Metrics saved to: {metrics_path}")
    
    logger.info("\nTesting completed!")


def main():
    parser = argparse.ArgumentParser(description="Test mJNet on 4D CTP data")
    
    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to directory containing 4D CTP NIfTI files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save predictions",
    )
    
    # Optional arguments
    parser.add_argument(
        "--label_dir",
        type=str,
        default=None,
        help="Path to directory containing label NIfTI files (for evaluation)",
    )
    
    # Model variant
    parser.add_argument("--longJ", action="store_true", help="Use longJ variant")
    parser.add_argument("--v2", action="store_true", help="Use v2 variant")
    
    # Inference parameters
    parser.add_argument("--stride", type=int, default=4, help="Stride for patch extraction")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference")
    
    args = parser.parse_args()
    
    # Create config
    config = Config()
    
    # Test
    test(config, args)


if __name__ == "__main__":
    main()
