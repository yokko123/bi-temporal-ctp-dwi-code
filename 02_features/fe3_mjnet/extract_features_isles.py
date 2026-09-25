#!/usr/bin/env python3
"""
Feature extraction from mJNet encoder bottleneck (PyTorch) — ISLES24.

Port of extract_features.py adapted to ISLES:
  - Patient IDs are 'sub-stroke<NNNN>' (single segment, no underscores).
  - CTP volumes live under
      <derivatives>/<pid>/ses-01/<pid>_ses-01_space-ncct_ctp.nii.gz
  - 6-class labels live in a FLAT folder:
      <label_dir>/<pid>_6class_map.nii.gz
    (ISLES24/analysis/ — union of 5- and 6-class outputs; cases without a
    CLB reference simply contain no label-5 voxels.)

The encoder/model side (checkpoints, encoder_forward) is unchanged — we are
applying the same SUS-trained mJNet folds to ISLES patches.

6-class mapping (0 = background, excluded):
    1 = Core  -> Final Infarct   (core_fi)
    2 = Core  -> Brain           (core_brain)
    3 = Penumbra -> Final Infarct (pen_fi)
    4 = Penumbra -> Brain        (pen_brain)
    5 = CLB   -> Brain           (clb_brain)
    6 = NHB   -> Final Infarct   (nhb_fi)

Output: CSV with columns
    patient, slice, y, x, region, feat_0 ... feat_255
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import SimpleITK as sitk

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from Architectures.arch_mJNet import create_mjnet

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---- 6-class mapping ----
CLASS_MAP = {
    1: "core_fi",
    2: "core_brain",
    3: "pen_fi",
    4: "pen_brain",
    5: "clb_brain",
    6: "nhb_fi",
}

# ---- Default paths ----
ISLES_BASE      = os.environ.get("ISLES24_WORK_ROOT", "/path/to/isles24-work")
DERIVATIVES_DIR = os.path.join(ISLES_BASE, "isles24_curated", "derivatives")
LABEL_DIR       = os.path.join(ISLES_BASE, "analysis")
CHECKPOINT_DIR  = os.environ.get("MJNET_CHECKPOINT_DIR", "/path/to/mjnet/checkpoints/cv_5fold")


# =====================================================================
#  Encoder-only forward pass
# =====================================================================
def encoder_forward(model, x):
    """
    Run only the encoder of standard mJNet and return the bottleneck
    feature vector (spatial-mean pooled).

    Args
        model : mJNet in eval mode
        x     : (B, 1, T, H, W)
    Returns
        (B, 256)
    """
    # --- Temporal collapse ---
    c = model.conv_1(x)
    c = F.relu(c)
    if model.batch_norm:
        c = model.conv_1_bn(c)
    c = F.avg_pool3d(c, kernel_size=(c.shape[2], 1, 1))     # (B, C, 1, H, W)

    # --- Encoder block 2 ---
    c = model.conv_2_1(c)
    c = model.conv_2_2(c)                                     # (B, 64, 1, H, W)
    c = F.max_pool3d(c, kernel_size=(1, 2, 2))                # (B, 64, 1, H/2, W/2)

    # --- Encoder block 3 ---
    c = model.conv_3_1(c)
    c = model.conv_3_2(c)                                     # (B, 128, 1, H/2, W/2)
    c = F.max_pool3d(c, kernel_size=(1, 2, 2))                # (B, 128, 1, H/4, W/4)

    # --- Bottleneck block 4 ---
    c = model.conv_4_1(c)
    c = model.conv_4_2(c)                                     # (B, 256, 1, 4, 4)

    # Spatial mean pooling → (B, 256)
    return c.squeeze(2).mean(dim=(2, 3))


# =====================================================================
#  Patient discovery (ISLES layout)
# =====================================================================
def discover_patients(derivatives_dir, label_dir):
    """
    Find ISLES patients that have both a CTP NIfTI and a 6-class label NIfTI.

    CTP layout:   <derivatives>/<pid>/ses-01/<pid>_ses-01_space-ncct_ctp.nii.gz
    Label layout: <label_dir>/<pid>_6class_map.nii.gz

    Returns list of (subj_name, patient_id, ctp_path, label_path).
    For ISLES, subj_name == patient_id (e.g. 'sub-stroke0001').
    """
    patients = []
    for subj in sorted(os.listdir(derivatives_dir)):
        if not subj.startswith("sub-stroke"):
            continue
        pid = subj  # ISLES: full subject name is the patient ID
        ctp_path   = os.path.join(derivatives_dir, subj, "ses-01",
                                  f"{subj}_ses-01_space-ncct_ctp.nii.gz")
        label_path = os.path.join(label_dir, f"{pid}_6class_map.nii.gz")
        if os.path.isfile(ctp_path) and os.path.isfile(label_path):
            patients.append((subj, pid, ctp_path, label_path))
    return patients


# =====================================================================
#  Model loading
# =====================================================================
def load_model(checkpoint_dir, fold, device):
    """Load a single fold model."""
    ckpt_path = os.path.join(checkpoint_dir, f"fold_{fold}", "checkpoint_best.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = create_mjnet(
        input_shape=(40, 16, 16),
        num_classes=4,
        batch_norm=True,
        dropout=True,
        dropout_rates={
            'long.1': 0.2, 'long.2': 0.2, 'long.3': 0.2,
            '1': 0.2, '2': 0.2, '3': 0.2, '4': 0.2, '5': 0.2,
        },
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    logging.info(f"Loaded fold_{fold} (epoch {ckpt.get('epoch', '?')})")
    return model


def load_models(checkpoint_dir, folds, device):
    """Load one or more fold models."""
    models = []
    for f in folds:
        models.append(load_model(checkpoint_dir, f, device))
    return models


# =====================================================================
#  Leak-aware fold assignment
# =====================================================================
def build_fold_assignment(checkpoint_dir, n_folds=5):
    """
    Recover the SUS CV split from the on-disk checkpoint layout:

        <checkpoint_dir>/fold_F/predictions/CTP_sub-stroke_<pid>/ -> SUS pid held out by fold F
        <checkpoint_dir>/test_predictions/CTP_sub-stroke_<pid>/   -> SUS pid held out from ALL folds

    Returns
        oof_map        : {pid -> fold_idx}   the fold that did NOT see this pid.
        test_patients  : set of pids in the global test set.

    Note for ISLES: the SUS patient IDs in those directories are 'NN_NNN'
    (e.g. '01_006'). ISLES patient IDs are 'sub-stroke<NNNN>' (single segment)
    so they cannot collide, and every ISLES patient is — by construction —
    unseen by all 5 folds. This map is therefore only used as a safety check.
    """
    PREFIX = "CTP_sub-stroke_"

    oof_map = {}
    for f in range(n_folds):
        pdir = os.path.join(checkpoint_dir, f"fold_{f}", "predictions")
        if not os.path.isdir(pdir):
            continue
        for name in os.listdir(pdir):
            if not name.startswith(PREFIX):
                continue
            pid = name[len(PREFIX):]
            if pid in oof_map and oof_map[pid] != f:
                logging.warning(
                    f"Patient {pid} held out by multiple folds "
                    f"({oof_map[pid]} and {f}); keeping fold {oof_map[pid]}."
                )
            else:
                oof_map[pid] = f

    test_dir = os.path.join(checkpoint_dir, "test_predictions")
    test_patients = set()
    if os.path.isdir(test_dir):
        for name in os.listdir(test_dir):
            if name.startswith(PREFIX):
                test_patients.add(name[len(PREFIX):])

    return oof_map, test_patients


# =====================================================================
#  Feature extraction for one patient
# =====================================================================
def extract_features_for_patient(
    models, ctp_path, label_path, patch_size=16, stride=1,
    device=torch.device("cpu"), batch_size=2048,
    fold_used_label=None,
):
    """
    Extract encoder bottleneck features for every labelled centre pixel.

    Returns a DataFrame with columns:
        y, x, region, slice, feat_0 … feat_255
    """
    # ---- Load CTP: (T, Z, Y, X) ----
    ctp = sitk.GetArrayFromImage(sitk.ReadImage(ctp_path)).astype(np.float32)
    T, Z, Y, X = ctp.shape

    # Normalise (same as training)
    ctp_norm = np.clip(ctp, 0.0, 255.0) / 255.0

    # ---- Load 6-class label: (Z, Y, X) ----
    label = sitk.GetArrayFromImage(sitk.ReadImage(label_path)).astype(np.int32)
    Z_lbl = label.shape[0]

    # Use minimum Z in case of slight mismatch
    Z_use = min(Z, Z_lbl)

    ps = patch_size
    c_off = ps // 2  # centre pixel offset

    feature_rows = []

    PENUMBRA_CLASSES = {3, 4}  # pen_fi, pen_brain

    for z in range(Z_use):
        label_slice = label[z]  # (Y, X)

        # Only process slices that contain penumbra labels
        has_penumbra = np.isin(label_slice, list(PENUMBRA_CLASSES)).any()
        if not has_penumbra:
            continue

        # Check that this slice has meaningful CTP signal
        ctp_slice_max = ctp_norm[:, z, :, :].max()
        if ctp_slice_max < 1e-3:
            logging.info(f"    slice {z:02d}: skipped (no CTP signal)")
            continue

        # Collect patches where centre pixel has a non-zero (1-6) label
        patches = []
        positions = []
        labels_list = []

        for y in range(0, Y - ps + 1, stride):
            cy = y + c_off
            for x in range(0, X - ps + 1, stride):
                cx = x + c_off
                lbl_val = label_slice[cy, cx]
                if lbl_val == 0:
                    continue

                patch = ctp_norm[:, z, y:y + ps, x:x + ps]  # (T, ps, ps)

                # Skip patches where centre pixel has no CTP signal
                centre_signal = ctp_norm[:, z, cy, cx]  # (T,)
                if centre_signal.max() < 1e-3:
                    continue

                # Skip patches with negligible overall CTP content
                if patch.mean() < 1e-3:
                    continue

                patches.append(patch)
                positions.append((cy, cx))
                labels_list.append(int(lbl_val))

        if not patches:
            continue

        patches_np = np.stack(patches)  # (N, T, ps, ps)

        # ---- Batch inference through encoder, ensemble-average ----
        all_features = []
        for i in range(0, len(patches_np), batch_size):
            batch_np = patches_np[i:i + batch_size]
            batch = torch.from_numpy(batch_np).float().unsqueeze(1).to(device)
            # (B, 1, T, ps, ps)

            fold_feats = []
            with torch.no_grad():
                for model in models:
                    feat = encoder_forward(model, batch)  # (B, 256)
                    fold_feats.append(feat)

            mean_feat = torch.stack(fold_feats).mean(dim=0)  # (B, 256)
            all_features.append(mean_feat.cpu().numpy())

        all_features = np.concatenate(all_features, axis=0)  # (N, 256)

        # Build rows
        for idx, ((cy, cx), lbl_val) in enumerate(zip(positions, labels_list)):
            row = {
                'y': cy,
                'x': cx,
                'region': CLASS_MAP.get(lbl_val, f"unknown_{lbl_val}"),
                'slice': z,
            }
            if fold_used_label is not None:
                row['fold_used'] = fold_used_label
            row.update({f'feat_{i}': float(v)
                        for i, v in enumerate(all_features[idx])})
            feature_rows.append(row)

        logging.info(f"    slice {z:02d}: {len(labels_list)} patches")

    return pd.DataFrame(feature_rows)


# =====================================================================
#  Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Extract mJNet encoder features with 6-class labels (ISLES24)")

    parser.add_argument("--derivatives_dir", type=str, default=DERIVATIVES_DIR)
    parser.add_argument("--label_dir", type=str, default=LABEL_DIR)
    parser.add_argument("--checkpoint_dir", type=str, default=CHECKPOINT_DIR)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="Which fold checkpoints to load (default: all 5). "
                             "Features are averaged across these folds.")
    parser.add_argument("--stride", type=int, default=1,
                        help="Patch extraction stride (1 = every pixel, "
                             "16 = non-overlapping)")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: auto-named)")

    # Leak-aware safety check (no functional change for ISLES — the SUS-trained
    # folds never saw ISLES patients, so the full ensemble is always safe.
    # This flag only triggers a verification + adds a `fold_used` column for
    # symmetry with extract_features.py).
    parser.add_argument("--leak_aware", dest="leak_aware",
                        action="store_true", default=True,
                        help="(default) Verify that no ISLES patient ID appears "
                             "in any SUS fold's training/val split; abort on "
                             "the first collision. ISLES IDs ('sub-stroke<NNNN>') "
                             "cannot collide with SUS IDs ('NN_NNN') by "
                             "construction, so this is just a sanity check.")
    parser.add_argument("--no_leak_aware", dest="leak_aware",
                        action="store_false",
                        help="Skip the leak-check (faster startup; same output).")

    # Chunking for large runs
    parser.add_argument("--chunks", type=int, default=1,
                        help="Split patients into N chunks")
    parser.add_argument("--chunk_idx", type=int, default=0,
                        help="0-based chunk index to process")

    args = parser.parse_args()

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    if device.type == "cuda":
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- Discover patients ----
    patients = discover_patients(args.derivatives_dir, args.label_dir)
    logging.info(f"Found {len(patients)} ISLES patients with CTP + 6-class labels")

    if not patients:
        logging.error("No patients found!")
        return

    # ---- Chunking ----
    chunks = np.array_split(patients, args.chunks)
    patients = list(chunks[args.chunk_idx])
    logging.info(f"Chunk {args.chunk_idx}/{args.chunks}: "
                 f"processing {len(patients)} patients")

    # ---- Load models ----
    models = load_models(args.checkpoint_dir, args.folds, device)
    logging.info(f"Loaded {len(models)} fold model(s): {args.folds}")

    # ---- Leak-aware safety check ----
    # ISLES patients should NEVER appear in any SUS fold's training/val split.
    # Different ID schemes ('sub-stroke<NNNN>' vs 'NN_NNN') guarantee this in
    # theory; we verify here to catch accidental data mixing.
    if args.leak_aware:
        oof_map, test_patients = build_fold_assignment(args.checkpoint_dir)
        seen_in_folds = set(oof_map.keys()) | test_patients
        isles_pids = {p[1] for p in patients}
        collisions = isles_pids & seen_in_folds
        if collisions:
            logging.error(
                f"LEAK DETECTED: {len(collisions)} ISLES patient IDs collide "
                f"with SUS fold splits: {sorted(collisions)[:10]}"
                f"{'...' if len(collisions) > 10 else ''}. Aborting."
            )
            return
        logging.info(
            f"Leak-aware check OK: 0 of {len(isles_pids)} ISLES patients "
            f"appear in any SUS fold split (SUS folds saw "
            f"{len(seen_in_folds)} patients total)."
        )

    if len(models) > 1:
        fold_used_label = f"ensemble_all_{'_'.join(map(str, args.folds))}"
    else:
        fold_used_label = f"single_{args.folds[0]}"

    # ---- Process patients ----
    all_dfs = []
    for i, (subj, pid, ctp_path, label_path) in enumerate(patients):
        logging.info(f"\n[{i + 1}/{len(patients)}] {subj}")
        t0 = time.time()

        try:
            df = extract_features_for_patient(
                models=models,
                ctp_path=ctp_path,
                label_path=label_path,
                patch_size=16,
                stride=args.stride,
                device=device,
                batch_size=args.batch_size,
                fold_used_label=fold_used_label,
            )
            if not df.empty:
                df['patient'] = pid
                all_dfs.append(df)
                logging.info(f"  {len(df)} feature rows "
                             f"({time.time() - t0:.1f}s)")
            else:
                logging.warning(f"  No features extracted (all labels empty?)")
        except Exception as e:
            logging.error(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    # ---- Save ----
    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)

        # Move patient & slice columns to front (include fold_used if present)
        front_cols = ['patient', 'slice', 'y', 'x', 'region']
        if 'fold_used' in df_all.columns:
            front_cols.append('fold_used')
        feat_cols = [c for c in df_all.columns if c.startswith('feat_')]
        df_all = df_all[front_cols + feat_cols]

        if args.output:
            out_path = args.output
        else:
            out_path = (f"encoder_features_isles_6class_stride{args.stride}"
                        f"_chunks{args.chunks}_idx{args.chunk_idx}_ensemble.csv")

        df_all.to_csv(out_path, index=False)
        logging.info(f"\nSaved {len(df_all)} rows to {out_path}")

        # Summary
        logging.info("\nPer-class counts:")
        logging.info(df_all['region'].value_counts().to_string())
    else:
        logging.warning("No features extracted from any patient.")


if __name__ == "__main__":
    main()
