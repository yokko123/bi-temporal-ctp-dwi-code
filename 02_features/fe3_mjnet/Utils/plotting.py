"""
Plotting utilities for mJNet training — EXACT match to reference TF code format.

Reference: CNN-segmentation-of-infarcted-regions/Model/testing.py::saveImage()

Output structure per patient/slice:
  predictions/
    CTP_PATIENT_ID/
      SLICE_INDEX.png           # Grayscale prediction [0, 85, 170, 255]
      HEATMAP/
        SLICE_INDEX_heatmap_penumbra.png  # Seaborn jet heatmap
        SLICE_INDEX_heatmap_core.png      # Seaborn jet heatmap
      GT/
        SLICE_INDEX.tiff          # Ground truth copy
      TMP/
        SLICE_INDEX.png           # Prediction with GT contours overlaid
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from typing import Dict, List, Optional, Tuple
from scipy import ndimage

# Visualization pixel values from original paper (Fig. 6c):
#   brain: [0, 60)  ->  pixel 0
#   penumbra: [60, 135)  ->  pixel 60
#   core: [135, 234)  ->  pixel 135
#   background: [234, 255]  ->  pixel 234
REF_PIXEL_VALUES = [234, 0, 60, 135]


# ======================================================================
#  Post-processing: remove small connected components
# ======================================================================
def postprocess_segmentation(
    seg: np.ndarray,
    min_component_size: int = 50,
    classes_to_clean: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Remove small connected components from a segmentation map.

    For each specified foreground class, connected components smaller than
    ``min_component_size`` voxels are replaced with the background class (0).
    Works on both 2-D slices and 3-D volumes.

    Parameters
    ----------
    seg : np.ndarray, shape (H, W) or (Z, H, W)
        Integer segmentation map with class indices.
    min_component_size : int
        Components with fewer voxels than this are removed.
    classes_to_clean : list of int or None
        Class indices to process.  Default: [1, 2, 3] (all foreground).

    Returns
    -------
    np.ndarray : Cleaned segmentation (same shape/dtype as input).
    """
    if classes_to_clean is None:
        classes_to_clean = [1, 2, 3]

    cleaned = seg.copy()
    for cls in classes_to_clean:
        binary = (cleaned == cls).astype(np.int32)
        if binary.sum() == 0:
            continue
        labelled, n_components = ndimage.label(binary)
        if n_components <= 1:
            # Only one component — keep if large enough, else remove
            if binary.sum() < min_component_size:
                cleaned[cleaned == cls] = 0
            continue
        # Measure component sizes
        component_sizes = ndimage.sum(binary, labelled, range(1, n_components + 1))
        for comp_idx, size in enumerate(component_sizes, start=1):
            if size < min_component_size:
                cleaned[labelled == comp_idx] = 0
    return cleaned
def plot_training_curves(
    history: Dict[str, List[float]],
    save_dir: str,
    filename: str = "training_curves.png",
) -> str:
    """Plot and save training curves in a single figure with 4 subplots."""
    os.makedirs(save_dir, exist_ok=True)

    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("mJNet Training Curves", fontsize=14, fontweight="bold")

    # 1) Loss
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train / Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2) Learning rate
    ax = axes[0, 1]
    ax.plot(epochs, history["lr"], "g-o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # 3) Per-class Dice
    ax = axes[1, 0]
    ax.plot(epochs, history["dice_brain"], "-", color="steelblue", label="Brain", linewidth=2)
    ax.plot(epochs, history["dice_penumbra"], "-", color="green", label="Penumbra", linewidth=2)
    ax.plot(epochs, history["dice_core"], "-", color="red", label="Core", linewidth=2)
    if "dice_background" in history:
        ax.plot(epochs, history["dice_background"], "--", color="gray", label="Background", alpha=0.4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice Coefficient")
    ax.set_title("Per-Class Dice (Validation)")
    ax.legend(fontsize=8)
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)

    # 4) Foreground mean dice (brain+pen+core) — single summary metric
    ax = axes[1, 1]
    fg_dice = history.get("foreground_dice", history.get("dice_mean", []))
    ax.plot(epochs, fg_dice, "m-", linewidth=2, label="Foreground Dice\n(brain+pen+core)")
    # Also show pen+core only (lesion dice) if we have per-class data
    if "dice_penumbra" in history and "dice_core" in history:
        lesion_dice = [(p + c) / 2.0
                       for p, c in zip(history["dice_penumbra"], history["dice_core"])]
        ax.plot(epochs, lesion_dice, "r--", linewidth=1.5, label="Lesion Dice\n(pen+core avg)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice Coefficient")
    ax.set_title("Summary Dice (Validation, excl. background)")
    ax.legend(fontsize=8)
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(save_dir, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ======================================================================
#  Prediction saving — EXACT MATCH to reference TF code
# ======================================================================
def save_prediction_slice(
    pred_label: np.ndarray,
    pred_probs: Optional[np.ndarray],
    gt_label: Optional[np.ndarray],
    patient_id: str,
    slice_idx: int,
    save_root: str,
):
    """
    Save prediction for a single slice in reference TF format.
    
    Parameters
    ----------
    pred_label : (H, W) int
        Predicted class indices [0, 1, 2, 3]
    pred_probs : (H, W, 4) float or None
        Softmax probabilities per class (optional, for heatmaps)
    gt_label : (H, W) int or None
        Ground truth class indices
    patient_id : str
        Patient ID (e.g., "01_008")
    slice_idx : int
        Slice index
    save_root : str
        Root save directory
    """
    # Create folder structure: CTP_PATIENT_ID/{HEATMAP, GT, TMP}/
    patient_folder = os.path.join(save_root, f"CTP_{patient_id}")
    heatmap_folder = os.path.join(patient_folder, "HEATMAP")
    gt_folder = os.path.join(patient_folder, "GT")
    tmp_folder = os.path.join(patient_folder, "TMP")
    
    for folder in [patient_folder, heatmap_folder, gt_folder, tmp_folder]:
        os.makedirs(folder, exist_ok=True)
    
    # Format slice index as 2-digit string
    idx_str = f"{slice_idx:02d}"
    
    # 1. Save main prediction as grayscale with pixel values [0, 85, 170, 255]
    pred_img = np.zeros_like(pred_label, dtype=np.uint8)
    for class_id, pixel_val in enumerate(REF_PIXEL_VALUES):
        pred_img[pred_label == class_id] = pixel_val
    
    pred_path = os.path.join(patient_folder, f"{idx_str}.png")
    cv2.imwrite(pred_path, pred_img)
    
    # 2. Save heatmaps for penumbra (class 2) and core (class 3) if probs available
    if pred_probs is not None:
        # Penumbra heatmap
        plt.figure(figsize=(6, 6))
        sns.heatmap(
            pred_probs[:, :, 2],
            cmap="jet",
            yticklabels=False,
            xticklabels=False,
            square=True,
            cbar=False,
        )
        penumbra_heatmap_path = os.path.join(heatmap_folder, f"{idx_str}_heatmap_penumbra.png")
        plt.savefig(penumbra_heatmap_path, transparent=True, bbox_inches="tight")
        plt.close()
        
        # Core heatmap
        plt.figure(figsize=(6, 6))
        sns.heatmap(
            pred_probs[:, :, 3],
            cmap="jet",
            yticklabels=False,
            xticklabels=False,
            square=True,
            cbar=False,
        )
        core_heatmap_path = os.path.join(heatmap_folder, f"{idx_str}_heatmap_core.png")
        plt.savefig(core_heatmap_path, transparent=True, bbox_inches="tight")
        plt.close()
    
    # 3. Save GT if provided
    if gt_label is not None:
        gt_img = np.zeros_like(gt_label, dtype=np.uint8)
        for class_id, pixel_val in enumerate(REF_PIXEL_VALUES):
            gt_img[gt_label == class_id] = pixel_val
        
        gt_path = os.path.join(gt_folder, f"{idx_str}.tiff")
        cv2.imwrite(gt_path, gt_img)
        
        # 4. Save TMP: prediction with GT contours overlaid
        pred_rgb = cv2.cvtColor(pred_img, cv2.COLOR_GRAY2RGB)
        
        # Penumbra contours (blue) - threshold to isolate penumbra pixels (60)
        penumbra_mask = ((gt_img >= 60) & (gt_img < 135)).astype(np.uint8) * 255
        penumbra_cnt, _ = cv2.findContours(
            penumbra_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        pred_rgb = cv2.drawContours(pred_rgb, penumbra_cnt, -1, (255, 0, 0), 2)
        
        # Core contours (red) - threshold to isolate core pixels (135)
        core_mask = ((gt_img >= 135) & (gt_img < 234)).astype(np.uint8) * 255
        core_cnt, _ = cv2.findContours(
            core_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        pred_rgb = cv2.drawContours(pred_rgb, core_cnt, -1, (0, 0, 255), 2)
        
        tmp_path = os.path.join(tmp_folder, f"{idx_str}.png")
        cv2.imwrite(tmp_path, pred_rgb)


def save_volume_predictions(
    model: nn.Module,
    volume_dataset: Dataset,
    device: torch.device,
    save_root: str,
    max_volumes: int = 5,
    skip_empty: bool = True,
    top_k_slices: int = 5,
    logger = None,
) -> int:
    """
    Save volume predictions in reference TF format for multiple patients.
    
    Creates folder structure:
    save_root/
      CTP_{patient_id}/
        {slice_idx:02d}.png                       # Grayscale prediction
        HEATMAP/{slice_idx:02d}_heatmap_*.png     # Jet heatmaps for penumbra/core
        GT/{slice_idx:02d}.tiff                   # Ground truth
        TMP/{slice_idx:02d}.png                   # Prediction with GT contours
    
    Parameters
    ----------
    model : nn.Module
        Trained model
    volume_dataset : Dataset
        CTPVolumeDataset with per-slice __getitem__
    device : torch.device
        Computation device
    save_root : str
        Root output folder
    max_volumes : int
        Max patients/slices to process
    skip_empty : bool
        If True, assign background class to empty patches without model inference
    top_k_slices : int
        Number of most interesting slices to save per patient (ranked by lesion content)
    logger : logging.Logger
        Optional logger for output messages
    
    Returns
    -------
    int : Number of volumes processed
    """
    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    model.eval()
    os.makedirs(save_root, exist_ok=True)
    
    dataset_len = len(volume_dataset)
    if dataset_len == 0:
        log("WARNING: Volume dataset is empty! No predictions to save.")
        return 0
    
    log(f"Dataset has {dataset_len} total slices")
    
    # Group slices by volume
    volume_slices = {}
    for idx in range(dataset_len):
        sample = volume_dataset[idx]
        vol_idx = sample['vol_idx']
        if vol_idx not in volume_slices:
            volume_slices[vol_idx] = []
        volume_slices[vol_idx].append(idx)
    
    log(f"Found {len(volume_slices)} unique volumes with {dataset_len} slices")
    
    processed_volumes = 0
    
    for vol_idx in sorted(volume_slices.keys()):
        if processed_volumes >= max_volumes:
            break
        
        slice_indices = volume_slices[vol_idx]
        sample = volume_dataset[slice_indices[0]]
        
        # Extract patient ID from CTP path
        # Filename: sub-stroke_01_001_ses-01_space-ncct_ctp.nii.gz
        # Patient ID: sub-stroke_01_001  (first 3 underscore-separated parts)
        ctp_path = sample['ctp_path']
        ctp_basename = os.path.basename(ctp_path).replace('.nii.gz', '').replace('.nii', '')
        parts = ctp_basename.split('_')
        if len(parts) >= 3:
            patient_id = f"{parts[0]}_{parts[1]}_{parts[2]}"  # "sub-stroke_01_001"
        elif len(parts) >= 2:
            patient_id = f"{parts[0]}_{parts[1]}"
        else:
            patient_id = f"vol_{vol_idx:03d}"
        
        log(f"Processing {patient_id} ({processed_volumes+1}/{max_volumes}): {len(slice_indices)} slices")
        
        # Storage for slice predictions and scores
        slice_scores = []
        slice_predictions = []
        
        for slice_dataset_idx in slice_indices:
            sample = volume_dataset[slice_dataset_idx]
            patches = sample['patches']  # (N, T, H, W)
            coords = sample['coords']    # [(y, x), ...]
            slice_shape = sample['slice_shape']  # (Y, X)
            slice_idx = sample['slice_idx']
            gt_label = sample.get('label', None)  # (Y, X) or None
            
            # Reconstruct full slice prediction
            pred_label, pred_probs = reconstruct_slice_from_patches(
                model, patches, coords, slice_shape, device, skip_empty=skip_empty
            )
            
            # Post-process: remove small connected components
            pred_label = postprocess_segmentation(pred_label, min_component_size=50)
            
            # Score slice by lesion content
            score = 0.0
            if gt_label is not None:
                gt_np = gt_label.numpy() if isinstance(gt_label, torch.Tensor) else gt_label
                penumbra_voxels = (gt_np == 2).sum()
                core_voxels = (gt_np == 3).sum()
                score = penumbra_voxels * 1.0 + core_voxels * 10.0
            else:
                penumbra_voxels = (pred_label == 2).sum()
                core_voxels = (pred_label == 3).sum()
                score = penumbra_voxels * 1.0 + core_voxels * 10.0
            
            slice_scores.append(score)
            slice_predictions.append((slice_idx, pred_label, pred_probs, gt_label))
        
        # Select top_k most interesting slices
        if len(slice_scores) == 0:
            log(f"  Warning: No slices found for {patient_id}")
            continue
        
        top_k = min(top_k_slices, len(slice_scores))
        top_indices = np.argsort(slice_scores)[::-1][:top_k]
        
        for rank, idx in enumerate(top_indices):
            slice_idx, pred_label, pred_probs, gt_label = slice_predictions[idx]
            
            # Convert gt_label to numpy if needed
            gt_np = None
            if gt_label is not None:
                gt_np = gt_label.numpy() if isinstance(gt_label, torch.Tensor) else gt_label
            
            save_prediction_slice(
                pred_label=pred_label,
                pred_probs=pred_probs,
                gt_label=gt_np,
                patient_id=patient_id,
                slice_idx=slice_idx,
                save_root=save_root,
            )
            log(f"  Saved slice {slice_idx:02d} (rank {rank+1}/{top_k}, score={slice_scores[idx]:.0f})")
        
        processed_volumes += 1
    
    log(f"Saved predictions for {processed_volumes} volumes to {save_root}")
    return processed_volumes


def reconstruct_slice_from_patches(
    model: nn.Module,
    patches: torch.Tensor,
    coords: List[Tuple[int, int]],
    slice_shape: Tuple[int, int],
    device: torch.device,
    skip_empty: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct full slice prediction from overlapping patches by
    averaging softmax probabilities where patches overlap.
    
    Parameters
    ----------
    model : nn.Module
        Trained model
    patches : (N, T, H, W) tensor
        Patch tensor
    coords : list of (y, x) tuples
        Patch coordinates (may overlap when stride < patch_size)
    slice_shape : (Y, X)
        Full slice dimensions
    device : torch.device
        Computation device
    skip_empty : bool
        Skip inference for empty patches
    
    Returns
    -------
    pred_label : (Y, X) ndarray
        Predicted class indices (argmax of averaged probabilities)
    pred_probs : (Y, X, 4) ndarray
        Averaged softmax probabilities
    """
    Y, X = slice_shape
    N, T, H, W = patches.shape
    num_classes = 4
    
    # Accumulate softmax probabilities and count overlaps
    prob_sum = np.zeros((Y, X, num_classes), dtype=np.float32)
    count_map = np.zeros((Y, X), dtype=np.float32)
    
    with torch.no_grad():
        # Batch inference for efficiency (process all patches at once or in chunks)
        batch_size = 512
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_patches = patches[start:end]  # (B, T, H, W)
            batch_coords = coords[start:end]
            
            # Find non-empty patches in this batch
            if skip_empty:
                patch_sums = batch_patches.abs().sum(dim=(1, 2, 3))  # (B,)
                non_empty_mask = patch_sums > 1e-6
            else:
                non_empty_mask = torch.ones(batch_patches.shape[0], dtype=torch.bool)
            
            # Handle empty patches: assign background probability
            for i in range(batch_patches.shape[0]):
                y, x = batch_coords[i]
                if not non_empty_mask[i]:
                    prob_sum[y:y+H, x:x+W, 0] += 1.0  # background
                    count_map[y:y+H, x:x+W] += 1.0
            
            # Run inference on non-empty patches
            non_empty_indices = torch.where(non_empty_mask)[0]
            if len(non_empty_indices) == 0:
                continue
            
            batch_input = batch_patches[non_empty_indices]  # (B', T, H, W)
            batch_input = batch_input.unsqueeze(1).float().to(device)  # (B', 1, T, H, W)
            
            logits = model(batch_input)  # (B', 4, H, W)
            probs = torch.softmax(logits, dim=1).cpu().numpy()  # (B', 4, H, W)
            
            for j, orig_idx in enumerate(non_empty_indices):
                y, x = batch_coords[orig_idx]
                prob_sum[y:y+H, x:x+W, :] += probs[j].transpose(1, 2, 0)  # (H, W, 4)
                count_map[y:y+H, x:x+W] += 1.0
    
    # Average probabilities where patches overlap
    count_map = np.maximum(count_map, 1e-8)  # avoid division by zero
    pred_probs = prob_sum / count_map[..., np.newaxis]
    
    # Final prediction from averaged probabilities
    pred_label = pred_probs.argmax(axis=-1).astype(np.int64)
    
    return pred_label, pred_probs


def predict_slice_with_patches(
    model: nn.Module,
    ctp_slice: np.ndarray,
    device: torch.device,
    patch_size: int = 16,
    skip_empty: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict a full CTP slice by dividing into patches.
    
    Parameters
    ----------
    model : nn.Module
        Trained model
    ctp_slice : (C, H, W, T) ndarray
        CTP slice data
    device : torch.device
        Computation device
    patch_size : int
        Patch size
    skip_empty : bool
        If True, assign background to empty patches without inference
    
    Returns
    -------
    pred_label : (H, W) ndarray
        Predicted class indices
    pred_probs : (H, W, 4) ndarray
        Softmax probabilities
    """
    C, H, W, T = ctp_slice.shape
    assert H % patch_size == 0 and W % patch_size == 0, "Image size must be divisible by patch_size"
    
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    
    # Initialize output
    pred_label = np.zeros((H, W), dtype=np.int64)
    pred_probs = np.zeros((H, W, 4), dtype=np.float32)
    
    with torch.no_grad():
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                y = i * patch_size
                x = j * patch_size
                
                patch = ctp_slice[:, y:y+patch_size, x:x+patch_size, :]  # (C, 16, 16, T)
                
                # Check if patch is empty (skip inference if requested)
                if skip_empty and np.abs(patch).sum() < 1e-6:
                    pred_label[y:y+patch_size, x:x+patch_size] = 0  # Background
                    pred_probs[y:y+patch_size, x:x+patch_size, 0] = 1.0
                    continue
                
                # Run inference
                patch_tensor = torch.from_numpy(patch).unsqueeze(0).float().to(device)  # (1, C, H, W, T)
                logits = model(patch_tensor)  # (1, 4, H, W)
                probs = torch.softmax(logits, dim=1)  # (1, 4, H, W)
                pred = probs.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)
                probs_np = probs.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 4)
                
                pred_label[y:y+patch_size, x:x+patch_size] = pred
                pred_probs[y:y+patch_size, x:x+patch_size, :] = probs_np
    
    return pred_label, pred_probs
