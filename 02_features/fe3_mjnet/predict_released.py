#!/usr/bin/env python3
"""
Standalone prediction script for the mJ-Net core/penumbra segmenter.

Loads the 5 fold checkpoints published in this repository, runs sliding-window
inference over a preprocessed admission CTP volume, and writes a 4-class label
map (0 background, 1 brain, 2 penumbra, 3 core).

The weights are released mainly as a FEATURE EXTRACTOR (FE3 in the paper).
Segmentation quality is low - penumbra Dice 0.30, core Dice 0.05 on the
held-out test set - so treat the label map as a coarse map, not a measurement.

For the 256-D encoder read-out that the paper actually analyses, use
02_features/fe3_mjnet/extract_features.py from the code repository. It pools
the bottleneck over 16x16 patches inside each labelled region, which is not
something this whole-volume script reproduces.

Input must be preprocessed exactly as in stage 01 of the code repository:
motion-corrected, resampled to 40 frames, registered to the admission NCCT,
skull-stripped, windowed to [0, 100] HU and rescaled to an 8-bit grey range.

Usage:
  # weights are fetched from the Hub on first run
  python predict.py --input CASE_ctp.nii.gz --output CASE_seg.nii.gz

  # 40 separate 3-D frames, nnU-Net style (CASE_0000.nii.gz ... CASE_0039.nii.gz)
  python predict.py --input CASE_dir/ --output CASE_seg.nii.gz

  # already have the weights
  python predict.py --input CASE_ctp.nii.gz --output CASE_seg.nii.gz --weights ./folds
"""

import argparse
import glob
import logging
import os
import sys

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

REPO_ID = "yokko123/ctp-core-penumbra-mjnet"
NUM_CLASSES = 4
CLASS_NAMES = ["background", "brain", "penumbra", "core"]
NUM_TIMEPOINTS = 40
PATCH_SIZE = 16


# =====================================================================
#  Weights
# =====================================================================
def resolve_weights(weights_dir):
    """Return a directory holding fold_0.pt ... fold_4.pt, downloading if needed."""
    if weights_dir:
        if not os.path.isdir(weights_dir):
            raise FileNotFoundError(f"--weights is not a directory: {weights_dir}")
        return weights_dir

    from huggingface_hub import snapshot_download
    logging.info(f"Fetching weights from {REPO_ID}")
    root = snapshot_download(repo_id=REPO_ID, allow_patterns=["folds/*.pt"])
    return os.path.join(root, "folds")


def load_models(weights_dir, folds, device):
    """Rebuild each fold from its own stored architecture description."""
    import torch
    # arch_mJNet.py sits next to this file in the Hub release, and under
    # Architectures/ in the code repository. Accept either.
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (here, os.path.join(here, "Architectures")):
        if os.path.isfile(os.path.join(cand, "arch_mJNet.py")):
            sys.path.insert(0, cand)
            break
    else:
        raise FileNotFoundError(
            f"arch_mJNet.py not found in {here} or {here}/Architectures")
    from arch_mJNet import create_mjnet

    models = []
    for f in folds:
        path = os.path.join(weights_dir, f"fold_{f}.pt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        arch = ckpt["arch"]
        model = create_mjnet(
            input_shape=tuple(arch["input_shape"]),
            num_classes=arch["num_classes"],
            batch_norm=arch["batch_norm"],
            dropout=arch["dropout"],
            dropout_rates=arch["dropout_rates"],
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device).eval()
        logging.info(f"Loaded fold_{f} (epoch {ckpt.get('epoch', '?')})")
        models.append(model)
    return models


# =====================================================================
#  I/O
# =====================================================================
def load_ctp(path):
    """Read a CTP case as (T, Z, Y, X), plus a reference image for geometry.

    Accepts a single 4-D NIfTI, or a directory of 40 3-D frames named
    <case>_0000.nii.gz ... <case>_0039.nii.gz.
    """
    import SimpleITK as sitk

    if os.path.isdir(path):
        frames = sorted(glob.glob(os.path.join(path, "*_[0-9][0-9][0-9][0-9].nii.gz")))
        if len(frames) != NUM_TIMEPOINTS:
            raise ValueError(
                f"Expected {NUM_TIMEPOINTS} frames in {path}, found {len(frames)}. "
                "Files must be named <case>_0000.nii.gz ... <case>_0039.nii.gz."
            )
        ref = sitk.ReadImage(frames[0])
        vol = np.stack([sitk.GetArrayFromImage(sitk.ReadImage(f)) for f in frames])
        return vol.astype(np.float32), ref

    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    if arr.ndim != 4:
        raise ValueError(
            f"{path} has shape {arr.shape}; expected a 4-D CTP volume or a "
            "directory of 40 3-D frames."
        )
    # SimpleITK gives (T, Z, Y, X) for a 4-D NIfTI whose 4th axis is time.
    if arr.shape[0] != NUM_TIMEPOINTS:
        raise ValueError(
            f"{path} has {arr.shape[0]} timepoints; this model requires "
            f"exactly {NUM_TIMEPOINTS}."
        )
    ref = img[:, :, :, 0]
    return arr.astype(np.float32), ref


def save_label_map(seg, ref, out_path):
    import SimpleITK as sitk
    out = sitk.GetImageFromArray(seg.astype(np.uint8))
    out.CopyInformation(ref)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    sitk.WriteImage(out, out_path)


# =====================================================================
#  Inference
# =====================================================================
def predict_volume(models, ctp, device, stride=4, batch_size=2048,
                   clip_min=0.0, clip_max=255.0):
    """Sliding-window 5-fold ensemble over a 4-D CTP volume.

    Returns (segmentation (Z,Y,X) uint8, mean probabilities (C,Z,Y,X) float32).
    Normalisation and patch handling match the paper's inference path exactly.
    """
    import torch
    import torch.nn.functional as F

    ctp_norm = (ctp.astype(np.float32) - clip_min) / (clip_max - clip_min + 1e-8)
    ctp_norm = np.clip(ctp_norm, 0.0, 1.0)

    T, Z, Y, X = ctp_norm.shape
    ps = PATCH_SIZE
    if Y < ps or X < ps:
        raise ValueError(f"Volume is {Y}x{X} in plane; needs at least {ps}x{ps}.")

    # Skip empty slices - the input is skull-stripped, so most volumes have them.
    valid_z = np.where(np.sum(np.abs(ctp_norm), axis=(0, 2, 3)) > 0)[0]

    prob_sum = np.zeros((NUM_CLASSES, Z, Y, X), dtype=np.float64)
    count = np.zeros((Z, Y, X), dtype=np.float64)

    ys = list(range(0, Y - ps + 1, stride))
    xs = list(range(0, X - ps + 1, stride))

    for mi, model in enumerate(models):
        logging.info(f"  fold {mi}: {len(valid_z)} slices x {len(ys) * len(xs)} patches")
        with torch.no_grad():
            for z in valid_z:
                positions = [(y, x) for y in ys for x in xs]
                patches = np.stack([ctp_norm[:, z, y:y + ps, x:x + ps]
                                    for y, x in positions])

                for i in range(0, len(patches), batch_size):
                    batch = torch.from_numpy(patches[i:i + batch_size]).float().to(device)
                    batch = batch.unsqueeze(1)                    # (B, 1, T, ps, ps)
                    probs = F.softmax(model(batch), dim=1).cpu().numpy()

                    for j, (y, x) in enumerate(positions[i:i + batch_size]):
                        prob_sum[:, z, y:y + ps, x:x + ps] += probs[j]
                        count[z, y:y + ps, x:x + ps] += 1

    mean_probs = (prob_sum / np.maximum(count, 1e-8)[np.newaxis]).astype(np.float32)
    return np.argmax(mean_probs, axis=0).astype(np.uint8), mean_probs


# =====================================================================
#  Main
# =====================================================================
def main():
    ap = argparse.ArgumentParser(
        description="mJ-Net core/penumbra prediction (5-fold ensemble).")
    ap.add_argument("--input", required=True,
                    help="4-D CTP NIfTI, or a directory of 40 3-D frames.")
    ap.add_argument("--output", required=True,
                    help="Output .nii.gz label map, or .csv with --features.")
    ap.add_argument("--weights", default=None,
                    help="Directory with fold_*.pt (default: download from the Hub).")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                    help="Folds to ensemble (default: all 5).")
    ap.add_argument("--stride", type=int, default=4,
                    help="Sliding-window stride; smaller is slower and smoother.")
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--device", default=None, help="cuda, cpu (default: auto).")
    ap.add_argument("--save-probs", default=None,
                    help="Optional .npy path for the mean softmax probabilities.")
    args = ap.parse_args()

    import torch
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logging.info(f"Device: {device}")

    weights_dir = resolve_weights(args.weights)
    models = load_models(weights_dir, args.folds, device)

    ctp, ref = load_ctp(args.input)
    logging.info(f"Input {args.input}: {ctp.shape} (T, Z, Y, X)")

    seg, probs = predict_volume(models, ctp, device,
                                stride=args.stride, batch_size=args.batch_size)

    save_label_map(seg, ref, args.output)
    logging.info(f"Wrote {args.output}")
    for v, name in enumerate(CLASS_NAMES):
        logging.info(f"  {name:<11} {int((seg == v).sum()):>10} voxels")

    if args.save_probs:
        np.save(args.save_probs, probs)
        logging.info(f"Wrote {args.save_probs}")


if __name__ == "__main__":
    main()
