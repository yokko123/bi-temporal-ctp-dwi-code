#!/usr/bin/env python3
"""extract_3D_ensemble_features.py

Extract encoder stage-3 features from the nnUNet 3D 5-fold ensemble
(Dataset1152_SUS_CTP_Reg_3DT, configuration 3d_fullres) for the 109 patients
listed in glcm_patient_max_with_clinical.csv.

Two output resolutions:
  --resolution native   (default)
      One row per stage-3 voxel that the model actually computes.
      Spatial scale 8x8 HW x 2 Z preprocessed voxels per row.
      Records both stage-3 indices (z_s,h_s,w_s) and approximate NCCT-center
      coordinates (z_ncct,h_ncct,w_ncct) so rows are comparable across patients.

  --resolution ncct
      Upsample features back to NCCT input resolution and emit one row per
      foreground voxel. Larger CSV; features are interpolated.

Aggregation: voxel CSV -> slice CSV -> patient CSV (mean/median/max pooling).
"""

import os, sys, glob, json, pickle, importlib, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import SimpleITK as sitk
import blosc2
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor

# =========================================================
# CONFIG
# =========================================================
DATASET_ROOT  = os.environ.get("NNUNET_DATASET_ROOT", "/path/to/nnUNetFrame/dataset")
DATASET_NAME  = "Dataset1152_SUS_CTP_Reg_3DT"
TRAINER_NAME  = "nnUNetTrainer__nnUNetPlans__3d_fullres"
CONFIGURATION = "3d_fullres"

CTP_DERIV_ROOT   = os.path.join(os.environ.get("SUS_CURATED_ROOT", "/path/to/Curated-SUS-2025"), "derivatives")
PATIENT_LIST_CSV = os.path.join(os.environ.get("FEATURE_WORK_ROOT", "/path/to/feature-work"), "3d_glcm", "updated_csv", "glcm_patient_max_with_clinical.csv")

PREPROC_DIR  = os.path.join(DATASET_ROOT, "nnUNet_preprocessed", DATASET_NAME, "nnUNetPlans_3d_fullres")
RAW_TS       = os.path.join(DATASET_ROOT, "nnUNet_raw", DATASET_NAME, "imagesTs")
MODEL_DIR    = os.path.join(DATASET_ROOT, "nnUNet_trained_models", DATASET_NAME, TRAINER_NAME)
PLANS_PATH   = os.path.join(MODEL_DIR, "plans.json")
DATASET_JSON = os.path.join(DATASET_ROOT, "nnUNet_raw", DATASET_NAME, "dataset.json")
SPLITS_PATH  = os.path.join(DATASET_ROOT, "nnUNet_preprocessed", DATASET_NAME, "splits_final.json")

OUT_DIR = os.path.join(os.environ.get("FEATURE_WORK_ROOT", "/path/to/feature-work"), "nnUnet_features_extraction", "features_3d_109")
os.makedirs(OUT_DIR, exist_ok=True)

INPUT_CHANNELS = 40
NUM_CLASSES    = 4
ENC_STAGE      = 3
FEATURE_DIM    = 256

# Cumulative encoder strides through stages 0..3 (computed from plans.json)
S3_STRIDES = (2, 8, 8)

CLASS6_NAMES = {
    1: "core_fi",
    2: "core_brain",
    3: "pen_fi",
    4: "pen_brain",
    5: "clb_brain",
    6: "nhb_fi",
}

FEAT_COLS = [f"f{i}" for i in range(FEATURE_DIM)]


# =========================================================
# MODEL
# =========================================================
def build_model(plans, device):
    from dynamic_network_architectures.architectures.unet import PlainConvUNet
    cfg  = plans["configurations"][CONFIGURATION]
    arch = dict(cfg["architecture"]["arch_kwargs"])
    for k in cfg["architecture"]["_kw_requires_import"]:
        if arch[k] is not None:
            mod, cls = arch[k].rsplit(".", 1)
            arch[k] = getattr(importlib.import_module(mod), cls)
    model = PlainConvUNet(input_channels=INPUT_CHANNELS, num_classes=NUM_CLASSES, **arch).to(device)
    model.eval()
    return model, arch, cfg


def load_fold_checkpoints():
    sds = []
    for fold in range(5):
        p = os.path.join(MODEL_DIR, f"fold_{fold}", "checkpoint_final.pth")
        ck = torch.load(p, map_location="cpu", weights_only=False)
        sds.append(ck["network_weights"])
    return sds


# =========================================================
# LEAK-AWARE FOLD ASSIGNMENT (mirrors the mJNet extractor)
# =========================================================
def build_oof_map(splits_path):
    """Recover the CV split from nnUNet's splits_final.json.

    For each fold, every case in its 'val' list was held out at training time
    (its out-of-fold / OOF fold).  Returns {case_id -> fold_idx}.

    Cases NOT present in any fold's train/val (the imagesTs hold-out set) are
    absent from this map; they were never seen by any fold, so they can safely
    use the full 5-fold ensemble.
    """
    with open(splits_path) as f:
        splits = json.load(f)
    oof = {}
    for fold_idx, sp in enumerate(splits):
        for case_id in sp.get("val", []):
            if case_id in oof and oof[case_id] != fold_idx:
                print(f"WARNING: {case_id} in val of multiple folds "
                      f"({oof[case_id]} and {fold_idx}); keeping {oof[case_id]}")
            else:
                oof[case_id] = fold_idx
    return oof


# =========================================================
# CASE LOADING
# =========================================================
def pid_to_case(pid):
    return "case_" + pid.split("sub-stroke_")[-1]

def is_train_case(case_id):
    return os.path.exists(os.path.join(PREPROC_DIR, f"{case_id}.b2nd"))

def load_train_case(case_id):
    img = blosc2.open(os.path.join(PREPROC_DIR, f"{case_id}.b2nd"))[:].astype(np.float32)
    with open(os.path.join(PREPROC_DIR, f"{case_id}.pkl"), "rb") as f:
        props = pickle.load(f)
    return img, props

def preprocess_test_case(case_id, plans_manager, configuration_manager, dataset_json):
    pp = DefaultPreprocessor(verbose=False)
    img_paths = sorted(glob.glob(os.path.join(RAW_TS, f"{case_id}_*.nii.gz")))
    if len(img_paths) != INPUT_CHANNELS:
        raise RuntimeError(f"{case_id}: expected {INPUT_CHANNELS} channel files, found {len(img_paths)}")
    data, _, props = pp.run_case(img_paths, None, plans_manager, configuration_manager, dataset_json)
    return data.astype(np.float32), props

def load_6class_mask(pid):
    p = os.path.join(CTP_DERIV_ROOT, pid, "ses-01", f"{pid}_space-ncct_bi-temporal_lesion_msk.nii.gz")
    if not os.path.exists(p):
        raise FileNotFoundError(f"6-class mask not found: {p}")
    return sitk.GetArrayFromImage(sitk.ReadImage(p)).astype(np.int32)


# =========================================================
# MASK PIPELINE: NCCT -> preprocessed -> stage-3 grid
# =========================================================
def resample_mask_to_preprocessed(mask_ncct, props, preprocessed_shape):
    """NCCT 6-class mask -> preprocessed grid (crop + nearest-neighbour HW resample).
    Z spacing is preserved (3.0) so Z just gets cropped."""
    bbox = props["bbox_used_for_cropping"]
    z_lo, z_hi = bbox[0]; y_lo, y_hi = bbox[1]; x_lo, x_hi = bbox[2]
    mask_crop = mask_ncct[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi]                  # (Z_p_crop, H_crop, W_crop)

    Zp, Hp, Wp = preprocessed_shape
    # If H,W differ from preprocessed (they will — different target spacing), resample with nearest-neighbour.
    if mask_crop.shape[1:] != (Hp, Wp):
        mt = torch.from_numpy(mask_crop.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1,1,Z,H,W)
        mt = F.interpolate(mt, size=mask_crop.shape[:1] + (Hp, Wp), mode='nearest')
        mask_crop = mt.squeeze(0).squeeze(0).numpy().astype(np.int32)

    # Z may also differ slightly — pad/crop in Z if needed (rare).
    if mask_crop.shape[0] != Zp:
        if mask_crop.shape[0] < Zp:
            pad = Zp - mask_crop.shape[0]
            mask_crop = np.pad(mask_crop, ((0, pad), (0, 0), (0, 0)), constant_values=0)
        else:
            mask_crop = mask_crop[:Zp]
    return mask_crop  # (Zp, Hp, Wp)


def block_pool_mask_to_stage3(mask_p, stride=S3_STRIDES, threshold=0.3):
    """Block-pool preprocessed mask to stage-3 grid using majority among non-zero labels.
    A stage-3 voxel gets label k>0 if foreground fraction in its block >= `threshold`
    AND k is the most frequent non-zero label there. Otherwise 0."""
    Z, H, W = mask_p.shape
    sz, sh, sw = stride
    Zs, Hs, Ws = Z // sz, H // sh, W // sw
    if Zs == 0 or Hs == 0 or Ws == 0:
        return np.zeros((max(Zs, 1), max(Hs, 1), max(Ws, 1)), dtype=np.int32)
    m = mask_p[:Zs*sz, :Hs*sh, :Ws*sw]
    block_size = sz * sh * sw

    # Per-class voxel counts in each block, vectorised.
    counts = np.zeros((7, Zs, Hs, Ws), dtype=np.int32)
    for cls in range(7):
        cm = (m == cls).astype(np.int32).reshape(Zs, sz, Hs, sh, Ws, sw)
        counts[cls] = cm.sum(axis=(1, 3, 5))

    fg_counts = counts[1:].sum(axis=0)                          # (Zs, Hs, Ws)
    fg_frac   = fg_counts.astype(np.float32) / block_size
    fg_arg    = counts[1:].argmax(axis=0) + 1                   # +1 because we sliced from class 1
    out = np.where(fg_frac >= threshold, fg_arg, 0).astype(np.int32)
    return out


# =========================================================
# COORDINATE MAPPING: stage-3 voxel -> NCCT center voxel
# =========================================================
def stage3_to_ncct_coords(z_s, h_s, w_s, props, preprocessed_shape, stride=S3_STRIDES):
    """Vectorised: given stage-3 indices arrays, return approximate NCCT coords."""
    bbox = props["bbox_used_for_cropping"]
    z_lo, z_hi = bbox[0]; y_lo, y_hi = bbox[1]; x_lo, x_hi = bbox[2]
    Zp, Hp, Wp = preprocessed_shape
    H_crop = y_hi - y_lo
    W_crop = x_hi - x_lo
    sz, sh, sw = stride

    z_p_center = sz * z_s + sz // 2
    h_p_center = sh * h_s + sh // 2
    w_p_center = sw * w_s + sw // 2

    z_ncct = z_lo + z_p_center                                  # Z spacing preserved
    h_ncct = y_lo + np.round(h_p_center * (H_crop / Hp)).astype(np.int32)
    w_ncct = x_lo + np.round(w_p_center * (W_crop / Wp)).astype(np.int32)
    return z_ncct, h_ncct, w_ncct


# =========================================================
# SLIDING-WINDOW INFERENCE WITH ENCODER HOOK
# =========================================================
def tile_positions(L, p, s):
    if L <= p:
        return [0]
    positions = list(range(0, L - p + 1, s))
    if positions[-1] != L - p:
        positions.append(L - p)
    return positions


def _pad_for_native(data, patch_size, stride):
    """Pad data so that:
      - each dim is at least patch_size
      - each dim is a multiple of the corresponding cumulative stride
    Returns padded data and the padded shape."""
    C, Z, H, W = data.shape
    pZ, pH, pW = patch_size
    sz, sh, sw = stride
    # Make at least patch_size and multiple-of-stride
    Z_target = max(Z, pZ)
    H_target = max(H, pH)
    W_target = max(W, pW)
    if Z_target % sz != 0: Z_target += (sz - Z_target % sz)
    if H_target % sh != 0: H_target += (sh - H_target % sh)
    if W_target % sw != 0: W_target += (sw - W_target % sw)
    pad_amt = [(0, 0), (0, Z_target - Z), (0, H_target - H), (0, W_target - W)]
    if any(p[1] > 0 for p in pad_amt):
        data = np.pad(data, pad_amt, mode='constant', constant_values=0.0)
    return data, (Z_target, H_target, W_target)


def run_sliding_window(model, data, fold_state_dicts, hook_cache,
                       patch_size, step_size, device, mode):
    """Sliding window inference with 5-fold ensemble at stage 3.

    mode == 'native': returns features at stage-3 grid (Z_p/sz, H_p/sh, W_p/sw).
    mode == 'ncct'  : returns features at preprocessed input grid (Z_p, H_p, W_p).
    """
    C, Z, H, W = data.shape
    pZ, pH, pW = patch_size
    sz, sh, sw = S3_STRIDES

    # Pad to multiples of cumulative stride so stage-3 placement is exact.
    data_padded, (Zp, Hp, Wp) = _pad_for_native(data, patch_size, S3_STRIDES)

    pZ_s, pH_s, pW_s = pZ // sz, pH // sh, pW // sw  # tile size at stage-3 grid

    if mode == 'native':
        Zs, Hs, Ws = Zp // sz, Hp // sh, Wp // sw
        feat_sum = torch.zeros((FEATURE_DIM, Zs, Hs, Ws), dtype=torch.float32, device=device)
        weight   = torch.zeros((1, Zs, Hs, Ws),           dtype=torch.float32, device=device)
    else:  # 'ncct'
        feat_sum = torch.zeros((FEATURE_DIM, Zp, Hp, Wp), dtype=torch.float32, device=device)
        weight   = torch.zeros((1, Zp, Hp, Wp),           dtype=torch.float32, device=device)

    zs = tile_positions(Zp, pZ, step_size[0])
    hs = tile_positions(Hp, pH, step_size[1])
    ws = tile_positions(Wp, pW, step_size[2])
    print(f"    tiles: {len(zs)}×{len(hs)}×{len(ws)} = {len(zs)*len(hs)*len(ws)}", flush=True)

    for z in zs:
        for h in hs:
            for w in ws:
                patch_np = data_padded[:, z:z+pZ, h:h+pH, w:w+pW]
                patch = torch.from_numpy(np.ascontiguousarray(patch_np)).unsqueeze(0).to(device).contiguous()
                f_acc = None
                for sd in fold_state_dicts:
                    model.load_state_dict(sd, strict=True)
                    with torch.no_grad():
                        _ = model(patch)
                    f = hook_cache["feat"]                              # (1, 256, pZ_s, pH_s, pW_s)
                    f_acc = f if f_acc is None else f_acc + f
                f_mean = (f_acc / len(fold_state_dicts)).squeeze(0)      # (256, pZ_s, pH_s, pW_s)

                if mode == 'native':
                    # Tile position must align with stride; verify (will hold given we padded to multiples).
                    assert z % sz == 0 and h % sh == 0 and w % sw == 0, \
                        f"tile pos ({z},{h},{w}) not aligned to stage-3 stride {S3_STRIDES}"
                    zsp, hsp, wsp = z // sz, h // sh, w // sw
                    feat_sum[:, zsp:zsp+pZ_s, hsp:hsp+pH_s, wsp:wsp+pW_s] += f_mean
                    weight[:,  zsp:zsp+pZ_s, hsp:hsp+pH_s, wsp:wsp+pW_s] += 1.0
                else:
                    f_up = F.interpolate(f_mean.unsqueeze(0), size=(pZ, pH, pW),
                                         mode='trilinear', align_corners=False).squeeze(0)
                    feat_sum[:, z:z+pZ, h:h+pH, w:w+pW] += f_up
                    weight[:,  z:z+pZ, h:h+pH, w:w+pW] += 1.0

                del patch, f_acc, f_mean
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    feat_avg = feat_sum / weight.clamp(min=1e-8)
    # Trim back to non-padded preprocessed shape (or its stage-3 equivalent).
    if mode == 'native':
        Zs_orig = (Z + sz - 1) // sz
        Hs_orig = (H + sh - 1) // sh
        Ws_orig = (W + sw - 1) // sw
        feat_avg = feat_avg[:, :Zs_orig, :Hs_orig, :Ws_orig].contiguous().cpu().numpy()
    else:
        feat_avg = feat_avg[:, :Z, :H, :W].contiguous().cpu().numpy()
    return feat_avg


# =========================================================
# NCCT MODE: per-slice inverse + voxel rows (kept from prior version)
# =========================================================
def yield_feat_slices_at_ncct(feat_avg, props):
    shape_orig  = props["shape_before_cropping"]
    bbox        = props["bbox_used_for_cropping"]
    Z_orig, H_orig, W_orig = shape_orig
    z_lo, z_hi = bbox[0]; y_lo, y_hi = bbox[1]; x_lo, x_hi = bbox[2]
    Z_p, H_p, W_p = feat_avg.shape[1:]
    H_crop, W_crop = y_hi - y_lo, x_hi - x_lo

    if Z_p != (z_hi - z_lo):
        ft = torch.from_numpy(feat_avg).unsqueeze(0)
        ft = F.interpolate(ft, size=(z_hi - z_lo, H_p, W_p),
                           mode='trilinear', align_corners=False)
        feat_avg = ft.squeeze(0).numpy()
        Z_p = z_hi - z_lo

    for z_p in range(Z_p):
        z_ncct = z_lo + z_p
        f_slice = torch.from_numpy(feat_avg[:, z_p]).unsqueeze(0)
        f_resamp = F.interpolate(f_slice, size=(H_crop, W_crop),
                                 mode='bilinear', align_corners=False).squeeze(0)
        pad = (x_lo, W_orig - x_hi, y_lo, H_orig - y_hi)
        f_full = F.pad(f_resamp, pad, mode='constant', value=0.0)
        yield z_ncct, f_full.numpy()


# =========================================================
# PER-PATIENT: write voxel rows
# =========================================================
def process_patient(pid, model, fold_sds, hook_cache,
                    plans_manager, config_manager, dataset_json,
                    patch_size, step_size, device, resolution,
                    out_handle, header_written, fold_used_label=None):
    case_id = pid_to_case(pid)
    train = is_train_case(case_id)
    split = "train" if train else "test"

    print(f"  [{split}] loading data ...", flush=True)
    if train:
        img, props = load_train_case(case_id)
    else:
        img, props = preprocess_test_case(case_id, plans_manager, config_manager, dataset_json)
    print(f"    image shape: {img.shape}  bbox: {props['bbox_used_for_cropping']}", flush=True)

    mask_ncct = load_6class_mask(pid)
    if tuple(mask_ncct.shape) != tuple(props["shape_before_cropping"]):
        raise RuntimeError(f"{pid}: mask shape {mask_ncct.shape} != orig {props['shape_before_cropping']}")

    print(f"    sliding window ({resolution}) ...", flush=True)
    feat_avg = run_sliding_window(model, img, fold_sds, hook_cache,
                                  patch_size, step_size, device, mode=resolution)

    rows_written = 0
    if resolution == 'native':
        # Build stage-3 mask
        mask_p   = resample_mask_to_preprocessed(mask_ncct, props, img.shape[1:])
        mask_s3  = block_pool_mask_to_stage3(mask_p, stride=S3_STRIDES, threshold=0.3)
        # mask_s3 may extend beyond feat_avg if rounding differs; align shapes
        Zs, Hs, Ws = feat_avg.shape[1:]
        m = mask_s3[:Zs, :Hs, :Ws]
        if m.shape != (Zs, Hs, Ws):
            # pad with zeros
            pad = [(0, max(0, t - s)) for s, t in zip(m.shape, (Zs, Hs, Ws))]
            m = np.pad(m, pad, constant_values=0)
        zsa, hsa, wsa = np.where((m >= 1) & (m <= 6))
        if len(zsa) > 0:
            labels = m[zsa, hsa, wsa]
            feats  = feat_avg[:, zsa, hsa, wsa].T            # (N, 256)
            z_ncct, h_ncct, w_ncct = stage3_to_ncct_coords(
                zsa, hsa, wsa, props, img.shape[1:], stride=S3_STRIDES
            )
            meta = pd.DataFrame({
                "patient_id": pid,
                "split":      split,
                "fold_used":  fold_used_label,
                "z_s":        zsa.astype(np.int32),
                "h_s":        hsa.astype(np.int32),
                "w_s":        wsa.astype(np.int32),
                "z_ncct":     z_ncct.astype(np.int32),
                "h_ncct":     h_ncct.astype(np.int32),
                "w_ncct":     w_ncct.astype(np.int32),
                "label":      labels.astype(np.int32),
                "class_name": [CLASS6_NAMES[int(l)] for l in labels],
            })
            feat_df = pd.DataFrame(feats.astype(np.float32), columns=FEAT_COLS)
            row_df  = pd.concat([meta, feat_df], axis=1)
            row_df.to_csv(out_handle, mode="a",
                          header=not header_written["done"],
                          index=False, float_format="%.5f")
            header_written["done"] = True
            rows_written += len(row_df)
    else:  # ncct mode
        for z_ncct, f_full in yield_feat_slices_at_ncct(feat_avg, props):
            m_slice = mask_ncct[z_ncct]
            ys, xs = np.where((m_slice >= 1) & (m_slice <= 6))
            if len(ys) == 0:
                continue
            labels = m_slice[ys, xs]
            feats  = f_full[:, ys, xs].T
            meta = pd.DataFrame({
                "patient_id": pid,
                "split":      split,
                "fold_used":  fold_used_label,
                "z":          z_ncct,
                "h":          ys,
                "w":          xs,
                "label":      labels,
                "class_name": [CLASS6_NAMES[int(l)] for l in labels],
            })
            feat_df = pd.DataFrame(feats.astype(np.float32), columns=FEAT_COLS)
            row_df  = pd.concat([meta, feat_df], axis=1)
            row_df.to_csv(out_handle, mode="a",
                          header=not header_written["done"],
                          index=False, float_format="%.5f")
            header_written["done"] = True
            rows_written += len(row_df)

    print(f"    rows: {rows_written:,}", flush=True)
    return rows_written


# =========================================================
# AGGREGATION
# =========================================================
def aggregate_csv(voxel_csv, slice_csv, patient_csv, resolution):
    """Read the voxel-level CSV and write slice + patient aggregations."""
    print(f"\nAggregating from {voxel_csv} ...", flush=True)
    df = pd.read_csv(voxel_csv)
    print(f"  total rows: {len(df):,}", flush=True)

    # Slice key: z_ncct (both modes have this — native renames z to z_ncct; ncct mode uses z).
    if "z_ncct" in df.columns:
        z_key = "z_ncct"
    else:
        z_key = "z"

    # Slice-level
    slice_rows = []
    for (pid, split, z, label, cname), g in df.groupby(["patient_id", "split", z_key, "label", "class_name"]):
        v = g[FEAT_COLS].values
        rec = {"patient_id": pid, "split": split, z_key: z,
               "label": label, "class_name": cname, "n_voxels": len(g)}
        for stat_name, stat_func in [("mean", np.mean), ("median", np.median), ("max", np.max)]:
            vec = stat_func(v, axis=0)
            for j, c in enumerate(FEAT_COLS):
                rec[f"{c}_{stat_name}"] = vec[j]
        slice_rows.append(rec)
    slice_df = pd.DataFrame(slice_rows)
    slice_df.to_csv(slice_csv, index=False, float_format="%.5f")
    print(f"  wrote slice-level CSV: {slice_csv}  ({len(slice_df):,} rows)", flush=True)

    # Patient-level (pool over slices using slice means)
    patient_rows = []
    base_cols = [f"{c}_mean" for c in FEAT_COLS]
    for (pid, split, label, cname), g in slice_df.groupby(["patient_id", "split", "label", "class_name"]):
        v = g[base_cols].values
        rec = {"patient_id": pid, "split": split, "label": label, "class_name": cname,
               "n_slices": len(g), "n_voxels_total": int(g["n_voxels"].sum())}
        for stat_name, stat_func in [("mean", np.mean), ("median", np.median), ("max", np.max)]:
            vec = stat_func(v, axis=0)
            for j, c in enumerate(FEAT_COLS):
                rec[f"{c}_{stat_name}"] = vec[j]
        patient_rows.append(rec)
    patient_df = pd.DataFrame(patient_rows)
    patient_df.to_csv(patient_csv, index=False, float_format="%.5f")
    print(f"  wrote patient-level CSV: {patient_csv}  ({len(patient_df):,} rows)", flush=True)


# =========================================================
# ENTRY POINT
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--resolution", default="native", choices=["native", "ncct"],
                    help="native: one row per stage-3 voxel (default). "
                         "ncct: upsample + one row per NCCT foreground voxel.")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end",   type=int, default=None)
    ap.add_argument("--skip-extract",   action="store_true")
    ap.add_argument("--skip-aggregate", action="store_true")

    # Leak-aware (out-of-fold) extraction — mirrors the mJNet extractor.
    ap.add_argument("--leak-aware", dest="leak_aware", action="store_true", default=True,
                    help="(default) For each TRAIN case, use ONLY the fold whose "
                         "validation split contained it (OOF). Hold-out test cases "
                         "(no preprocessed .b2nd) use the full 5-fold ensemble since "
                         "no fold saw them. Split recovered from splits_final.json.")
    ap.add_argument("--no-leak-aware", dest="leak_aware", action="store_false",
                    help="Disable leak-aware extraction. Average features across all "
                         "5 folds for every patient (previous behaviour; train cases "
                         "leak because 4/5 folds were trained on them).")
    ap.add_argument("--allow-unknown-train", action="store_true",
                    help="In leak-aware mode, if a train case is not found in any "
                         "fold's val split, fall back to the full ensemble (and warn) "
                         "instead of skipping.")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")
    print(f"Resolution mode: {args.resolution}")
    torch.backends.cudnn.benchmark = True

    mode_tag = "oof" if args.leak_aware else "ensemble"
    voxel_csv   = os.path.join(OUT_DIR, f"voxel_features_109patients_{mode_tag}_{args.resolution}.csv")
    slice_csv   = os.path.join(OUT_DIR, f"slice_features_109patients_{mode_tag}_{args.resolution}.csv")
    patient_csv = os.path.join(OUT_DIR, f"patient_features_109patients_{mode_tag}_{args.resolution}.csv")
    print(f"Output:\n  {voxel_csv}\n  {slice_csv}\n  {patient_csv}")

    with open(PLANS_PATH) as f:   plans = json.load(f)
    with open(DATASET_JSON) as f: dataset_json = json.load(f)
    cfg = plans["configurations"][CONFIGURATION]
    PATCH_SIZE = tuple(cfg["patch_size"])
    STEP = tuple(max(1, p // 2) for p in PATCH_SIZE)
    print(f"patch_size={PATCH_SIZE}  step={STEP}  S3_strides={S3_STRIDES}")

    model, arch_kwargs, _ = build_model(plans, device)
    fold_sds = load_fold_checkpoints()
    print(f"Loaded {len(fold_sds)} fold checkpoints")

    # ---- Leak-aware fold assignment ----
    if args.leak_aware:
        oof_map = build_oof_map(SPLITS_PATH)
        print(f"Leak-aware mode: OOF assignment for {len(oof_map)} train cases; "
              f"hold-out test cases use the full 5-fold ensemble.")
    else:
        oof_map = {}
        print("WARNING: leak-aware DISABLED — features for every patient are averaged "
              "across all 5 folds. Train cases leak (4/5 folds were trained on them).")

    hook_cache = {}
    def _enc_hook(module, inp, out):
        hook_cache["feat"] = out.detach()
    model.encoder.stages[ENC_STAGE].register_forward_hook(_enc_hook)

    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    plans_manager  = PlansManager(plans)
    config_manager = plans_manager.get_configuration(CONFIGURATION)

    df_pat = pd.read_csv(PATIENT_LIST_CSV)
    patients = sorted(df_pat["patient_id"].dropna().unique().tolist())
    end = args.end if args.end is not None else len(patients)
    patients = patients[args.start:end]
    print(f"Processing {len(patients)} patients (start={args.start}, end={end})")

    if not args.skip_extract:
        mode_open = "a" if args.start > 0 and os.path.exists(voxel_csv) else "w"
        out_handle = open(voxel_csv, mode_open, newline="")
        header_written = {"done": (mode_open == "a")}
        try:
            for i, pid in enumerate(patients, 1):
                print(f"\n[{i}/{len(patients)}] {pid}", flush=True)

                # ---- Decide which fold model(s) to use for this patient ----
                case_id = pid_to_case(pid)
                if args.leak_aware:
                    if case_id in oof_map:
                        f_idx = oof_map[case_id]
                        sds_to_use = [fold_sds[f_idx]]
                        fold_used_label = f"oof_{f_idx}"
                    elif is_train_case(case_id):
                        # Train case but missing from every val split (shouldn't happen).
                        if args.allow_unknown_train:
                            sds_to_use = fold_sds
                            fold_used_label = "ensemble_unknown"
                            print(f"  {case_id}: train case not in any val split; "
                                  f"full ensemble (LEAK RISK)", flush=True)
                        else:
                            print(f"  {case_id}: train case not in any val split; "
                                  f"SKIPPING (use --allow-unknown-train to ensemble)",
                                  flush=True)
                            continue
                    else:
                        # Genuine hold-out test case — no fold saw it.
                        sds_to_use = fold_sds
                        fold_used_label = "ensemble_test"
                else:
                    sds_to_use = fold_sds
                    fold_used_label = "ensemble_all"

                print(f"  using: {fold_used_label} "
                      f"({len(sds_to_use)} fold model(s))", flush=True)

                try:
                    process_patient(pid, model, sds_to_use, hook_cache,
                                    plans_manager, config_manager, dataset_json,
                                    PATCH_SIZE, STEP, device, args.resolution,
                                    out_handle, header_written,
                                    fold_used_label=fold_used_label)
                except Exception as e:
                    print(f"  FAILED: {e}", flush=True)
                    import traceback; traceback.print_exc()
        finally:
            out_handle.close()
        print(f"\nExtraction complete -> {voxel_csv}")

    if not args.skip_aggregate:
        aggregate_csv(voxel_csv, slice_csv, patient_csv, args.resolution)


if __name__ == "__main__":
    main()
