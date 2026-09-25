"""
Prediction script for mJNet with 5-fold cross-validation ensemble.

Loads all fold checkpoints, runs inference on each volume, and optionally
ensembles softmax probabilities across folds for the final segmentation.

Supports:
  - Curated BIDS-like derivatives_dir layout
  - Single-fold or multi-fold ensemble predictions
  - NIfTI output with same geometry as input
  - Per-patient + aggregate Dice/IoU metrics when labels are available

Usage:
  python predict.py \\
    --cv_dir checkpoints/cv_5fold_20260310_174037 \\
    --derivatives_dir $SUS_CURATED_ROOT \\
    --output_dir predictions/ \\
    --patients sub-stroke_01_001 sub-stroke_01_002 \\
    --stride 4
"""

import os
import sys
import re
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from Architectures.arch_mJNet import create_mjnet


# ======================================================================
#  Logging
# ======================================================================

def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"predict_{timestamp}.log")

    logger = logging.getLogger("predict")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        sh = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(fmt)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


# ======================================================================
#  Model loading
# ======================================================================

def load_fold_models(
    cv_dir: str,
    num_classes: int = 4,
    timepoints: int = 40,
    patch_size: int = 16,
    device: torch.device = torch.device("cpu"),
    logger: Optional[logging.Logger] = None,
) -> List[nn.Module]:
    """Load best checkpoint from each fold directory.

    Expects layout:
        cv_dir/fold_0/checkpoint_best.pt
        cv_dir/fold_1/checkpoint_best.pt
        ...
    """
    fold_dirs = sorted(
        [d for d in os.listdir(cv_dir) if d.startswith("fold_")],
        key=lambda d: int(d.split("_")[1]),
    )
    if not fold_dirs:
        raise FileNotFoundError(f"No fold_* directories found in {cv_dir}")

    models = []
    for fd in fold_dirs:
        ckpt_path = os.path.join(cv_dir, fd, "checkpoint_best.pt")
        if not os.path.exists(ckpt_path):
            if logger:
                logger.warning(f"Skipping {fd}: no checkpoint_best.pt")
            continue

        model = create_mjnet(
            input_shape=(timepoints, patch_size, patch_size),
            num_classes=num_classes,
            dropout=False,  # inference mode — no dropout
        )
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        models.append(model)

        if logger:
            epoch = ckpt.get("epoch", "?")
            logger.info(f"  Loaded {fd}/checkpoint_best.pt (epoch {epoch})")

    if logger:
        logger.info(f"  Total models loaded: {len(models)}")
    return models


# ======================================================================
#  Patient discovery
# ======================================================================

def discover_patients(
    derivatives_dir: str,
    patient_ids: Optional[List[str]] = None,
) -> List[Tuple[str, str, str]]:
    """Discover (patient_id, ctp_path, label_path) from curated layout.

    If patient_ids is None, returns ALL patients found.  Labels may not exist
    (label_path will be None in that case).
    """
    deriv_root = os.path.join(derivatives_dir, "derivatives")
    if not os.path.isdir(deriv_root):
        raise FileNotFoundError(f"No 'derivatives' folder in {derivatives_dir}")

    patients = []
    for subj in sorted(os.listdir(deriv_root)):
        if not re.match(r"sub-stroke_\d{2}_\d{3}$", subj):
            continue
        if patient_ids is not None and subj not in patient_ids:
            continue

        ses_dir = os.path.join(deriv_root, subj, "ses-01")
        ctp_path = os.path.join(ses_dir, f"{subj}_ses-01_space-ncct_ctp.nii.gz")
        lbl_path = os.path.join(ses_dir, f"{subj}_ses-01_space-ncct_ctp_lesion_msk_3class.nii.gz")

        if not os.path.isfile(ctp_path):
            continue

        patients.append((
            subj,
            ctp_path,
            lbl_path if os.path.isfile(lbl_path) else None,
        ))

    return patients


# ======================================================================
#  Volume-level inference
# ======================================================================

def predict_volume(
    models: List[nn.Module],
    ctp: np.ndarray,
    device: torch.device,
    patch_size: int = 16,
    stride: int = 4,
    batch_size: int = 2048,
    clip_min: float = 0.0,
    clip_max: float = 255.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run ensemble inference on a full 4D CTP volume.

    Args:
        models:     List of trained fold models.
        ctp:        Raw CTP volume, shape (T, Z, Y, X) float32.
        device:     Torch device.
        patch_size: Spatial patch size.
        stride:     Inference stride (smaller = more overlap = better but slower).
        batch_size: Patches per GPU batch.
        clip_min/max: Normalisation range (same as training).

    Returns:
        segmentation: (Z, Y, X) int16 — argmax class per voxel.
        probabilities: (C, Z, Y, X) float32 — mean softmax probabilities.
    """
    # Normalise (same as training)
    ctp_norm = (ctp.astype(np.float32) - clip_min) / (clip_max - clip_min + 1e-8)
    ctp_norm = np.clip(ctp_norm, 0.0, 1.0)

    T, Z, Y, X = ctp_norm.shape
    C = 4  # num_classes
    ps = patch_size

    # Find valid slices (non-zero CTP)
    slice_sums = np.sum(np.abs(ctp_norm), axis=(0, 2, 3))
    valid_z = np.where(slice_sums > 0)[0]

    # Accumulator: sum of softmax probs across models
    prob_sum = np.zeros((C, Z, Y, X), dtype=np.float64)
    count = np.zeros((Z, Y, X), dtype=np.float64)

    # Build patch grid
    ys = list(range(0, Y - ps + 1, stride))
    xs = list(range(0, X - ps + 1, stride))

    for model in models:
        with torch.no_grad():
            for z in valid_z:
                # Extract all patches for this slice
                patches = []
                positions = []
                for y in ys:
                    for x in xs:
                        patches.append(ctp_norm[:, z, y:y+ps, x:x+ps])
                        positions.append((y, x))

                if not patches:
                    continue

                patches_np = np.stack(patches)  # (N, T, ps, ps)

                # Process in batches
                for i in range(0, len(patches_np), batch_size):
                    batch = torch.from_numpy(patches_np[i:i+batch_size]).float().to(device)
                    batch = batch.unsqueeze(1)  # (B, 1, T, ps, ps)

                    logits = model(batch)  # (B, C, ps, ps)
                    probs = F.softmax(logits, dim=1).cpu().numpy()

                    for j, (y, x) in enumerate(positions[i:i+batch_size]):
                        prob_sum[:, z, y:y+ps, x:x+ps] += probs[j]
                        count[z, y:y+ps, x:x+ps] += 1

    # Average over models × overlap
    count = np.maximum(count, 1e-8)
    mean_probs = (prob_sum / count[np.newaxis]).astype(np.float32)

    segmentation = np.argmax(mean_probs, axis=0).astype(np.int16)

    return segmentation, mean_probs


# ======================================================================
#  Evaluation
# ======================================================================

def compute_metrics(pred: np.ndarray, gt: np.ndarray, num_classes: int = 4) -> Dict[str, float]:
    """Compute per-class Dice and IoU + lesion Dice."""
    class_names = ["background", "brain", "penumbra", "core"]
    metrics = {}

    for c in range(num_classes):
        p = (pred == c).astype(np.float64)
        g = (gt == c).astype(np.float64)
        inter = (p * g).sum()
        union = p.sum() + g.sum()
        metrics[f"dice_{class_names[c]}"] = float((2 * inter + 1e-8) / (union + 1e-8))
        metrics[f"iou_{class_names[c]}"] = float((inter + 1e-8) / (union - inter + 1e-8))

    # Foreground mean (brain + penumbra + core)
    metrics["dice_foreground"] = np.mean([metrics[f"dice_{class_names[c]}"] for c in range(1, num_classes)])

    # Lesion (penumbra + core combined)
    p_les = ((pred == 2) | (pred == 3)).astype(np.float64)
    g_les = ((gt == 2) | (gt == 3)).astype(np.float64)
    inter = (p_les * g_les).sum()
    union = p_les.sum() + g_les.sum()
    metrics["dice_lesion"] = float((2 * inter + 1e-8) / (union + 1e-8))

    return metrics


# ======================================================================
#  Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run mJNet ensemble predictions from CV fold checkpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict specific patients:
  python predict.py --cv_dir checkpoints/cv_5fold_20260310_174037 \\
    --derivatives_dir $SUS_CURATED_ROOT \\
    --patients sub-stroke_01_001 sub-stroke_01_002

  # Predict ALL patients (no --patients flag):
  python predict.py --cv_dir checkpoints/cv_5fold_20260310_174037 \\
    --derivatives_dir $SUS_CURATED_ROOT

  # Single-fold prediction (no ensemble):
  python predict.py --cv_dir checkpoints/cv_5fold_20260310_174037 \\
    --folds 0 --stride 4
        """,
    )

    parser.add_argument("--cv_dir", type=str, required=True,
                        help="Path to CV run directory (contains fold_0/, fold_1/, ...)")
    parser.add_argument("--derivatives_dir", type=str,
                        default=os.environ.get("SUS_CURATED_ROOT", "/path/to/Curated-SUS-2025"),
                        help="Path to curated dataset root")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: {cv_dir}/predictions)")
    parser.add_argument("--patients", type=str, nargs="*", default=None,
                        help="Specific patient IDs to predict. If omitted, predicts ALL.")
    parser.add_argument("--folds", type=int, nargs="*", default=None,
                        help="Fold indices to use (e.g., --folds 0 2 4). Default: all folds.")
    parser.add_argument("--stride", type=int, default=4,
                        help="Inference stride (default: 4). Smaller = denser overlap = better.")
    parser.add_argument("--batch_size", type=int, default=2048,
                        help="Patches per GPU batch (default: 2048)")
    parser.add_argument("--save_probs", action="store_true",
                        help="Save softmax probability maps as NIfTI (large files)")
    parser.add_argument("--no_eval", action="store_true",
                        help="Skip evaluation even if labels exist")

    args = parser.parse_args()

    # Defaults
    if args.output_dir is None:
        args.output_dir = os.path.join(args.cv_dir, "predictions")
    os.makedirs(args.output_dir, exist_ok=True)

    logger = setup_logging(args.output_dir)
    logger.info("=" * 60)
    logger.info("  mJNet Ensemble Prediction")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- Load models ---
    logger.info(f"\nLoading models from: {args.cv_dir}")

    # If specific folds requested, filter
    if args.folds is not None:
        # Temporarily rename non-selected folds so loader skips them
        all_models = load_fold_models(args.cv_dir, device=device, logger=logger)
        models = [all_models[i] for i in args.folds if i < len(all_models)]
        logger.info(f"Using folds: {args.folds} ({len(models)} models)")
    else:
        models = load_fold_models(args.cv_dir, device=device, logger=logger)
        logger.info(f"Using all {len(models)} fold models (ensemble)")

    if not models:
        logger.error("No models loaded! Check --cv_dir path.")
        return

    # --- Discover patients ---
    logger.info(f"\nDiscovering patients from: {args.derivatives_dir}")
    patients = discover_patients(args.derivatives_dir, args.patients)
    logger.info(f"Found {len(patients)} patients to process")

    if not patients:
        logger.error("No patients found! Check --derivatives_dir and --patients args.")
        return

    # --- Run predictions ---
    all_metrics = []
    config = Config()

    for idx, (pid, ctp_path, lbl_path) in enumerate(patients):
        logger.info(f"\n[{idx+1}/{len(patients)}] {pid}")
        logger.info(f"  CTP: {ctp_path}")

        # Load CTP via SimpleITK → (T, Z, Y, X)
        ctp_sitk = sitk.ReadImage(ctp_path)
        ctp = sitk.GetArrayFromImage(ctp_sitk).astype(np.float32)
        logger.info(f"  Shape: {ctp.shape} (T={ctp.shape[0]}, Z={ctp.shape[1]})")

        # Run ensemble prediction
        segmentation, probs = predict_volume(
            models=models,
            ctp=ctp,
            device=device,
            patch_size=config.data.patch_size,
            stride=args.stride,
            batch_size=args.batch_size,
            clip_min=config.data.clip_min,
            clip_max=config.data.clip_max,
        )

        # --- Save segmentation as NIfTI ---
        # Use original CTP geometry but only Z,Y,X (3D label)
        # Read the label to get the correct 3D reference geometry
        if lbl_path is not None:
            ref_sitk = sitk.ReadImage(lbl_path)
        else:
            # Create 3D reference from 4D CTP: extract first timepoint's geometry
            ref_size = list(ctp_sitk.GetSize())[:3]
            ref_spacing = list(ctp_sitk.GetSpacing())[:3]
            ref_origin = list(ctp_sitk.GetOrigin())[:3]
            ref_direction = ctp_sitk.GetDirection()
            # Extract 3x3 direction from 4x4
            dir_4x4 = np.array(ref_direction).reshape(4, 4) if len(ref_direction) == 16 else None
            if dir_4x4 is not None:
                dir_3x3 = dir_4x4[:3, :3].flatten().tolist()
            else:
                dir_3x3 = list(ref_direction[:9])
            ref_sitk = sitk.Image(ref_size, sitk.sitkInt16)
            ref_sitk.SetSpacing(ref_spacing)
            ref_sitk.SetOrigin(ref_origin)
            ref_sitk.SetDirection(dir_3x3)

        # segmentation is (Z, Y, X) — same axes as SimpleITK internal layout
        seg_sitk = sitk.GetImageFromArray(segmentation.astype(np.int16))
        seg_sitk.CopyInformation(ref_sitk)

        out_path = os.path.join(args.output_dir, f"{pid}_prediction.nii.gz")
        sitk.WriteImage(seg_sitk, out_path)
        logger.info(f"  Saved: {out_path}")

        # Count classes in prediction
        unique, counts = np.unique(segmentation, return_counts=True)
        for u, c in zip(unique, counts):
            pct = 100 * c / segmentation.size
            logger.info(f"    Class {u}: {c:,} voxels ({pct:.2f}%)")

        # --- Optionally save probability maps ---
        if args.save_probs:
            prob_path = os.path.join(args.output_dir, f"{pid}_probs.nii.gz")
            # probs is (C, Z, Y, X) → save as 4D
            prob_sitk = sitk.GetImageFromArray(probs)  # SimpleITK treats first axis as component
            sitk.WriteImage(prob_sitk, prob_path)
            logger.info(f"  Saved probs: {prob_path}")

        # --- Evaluate ---
        if lbl_path is not None and not args.no_eval:
            gt = sitk.GetArrayFromImage(sitk.ReadImage(lbl_path)).astype(np.int16)
            metrics = compute_metrics(segmentation, gt)
            all_metrics.append((pid, metrics))

            logger.info(f"  Dice — Brain: {metrics['dice_brain']:.4f}, "
                        f"Penumbra: {metrics['dice_penumbra']:.4f}, "
                        f"Core: {metrics['dice_core']:.4f}")
            logger.info(f"  Foreground Dice: {metrics['dice_foreground']:.4f}, "
                        f"Lesion Dice: {metrics['dice_lesion']:.4f}")

    # --- Summary ---
    if all_metrics:
        logger.info("\n" + "=" * 60)
        logger.info("  Aggregate Metrics")
        logger.info("=" * 60)

        metric_keys = list(all_metrics[0][1].keys())
        for key in metric_keys:
            values = [m[key] for _, m in all_metrics]
            logger.info(f"  {key:<25s}  {np.mean(values):.4f} ± {np.std(values):.4f}")

        # Save CSV
        csv_path = os.path.join(args.output_dir, "metrics.csv")
        with open(csv_path, "w") as f:
            f.write("patient," + ",".join(metric_keys) + "\n")
            for pid, m in all_metrics:
                vals = ",".join(f"{m[k]:.6f}" for k in metric_keys)
                f.write(f"{pid},{vals}\n")
            # Summary row
            f.write("MEAN," + ",".join(f"{np.mean([m[k] for _, m in all_metrics]):.6f}" for k in metric_keys) + "\n")
            f.write("STD," + ",".join(f"{np.std([m[k] for _, m in all_metrics]):.6f}" for k in metric_keys) + "\n")

        logger.info(f"\nMetrics saved to: {csv_path}")

    logger.info(f"\nAll predictions saved to: {args.output_dir}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
