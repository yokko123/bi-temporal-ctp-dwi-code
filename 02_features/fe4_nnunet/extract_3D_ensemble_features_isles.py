#!/usr/bin/env python3
"""extract_3D_ensemble_features_isles.py

Extract encoder stage-3 features from the SUS-trained nnUNet 3D 5-fold ensemble
(Dataset1152_SUS_CTP_Reg_3DT, configuration 3d_fullres) applied to the ISLES24
CTP test inputs (Dataset1155_ISLES24_CTP_3DMultiChannel, imagesTs/).

Inputs:
  - Raw CTP volumes: nnUNet_raw/Dataset1155_ISLES24_CTP_3DMultiChannel/imagesTs/
      filenames: case_NNNN_0000.nii.gz ... case_NNNN_0039.nii.gz (40 channels)
  - 6-class masks:   <ISLES_BASE>/analysis/sub-strokeNNNN_6class_map.nii.gz
  - Model weights:   nnUNet_trained_models/Dataset1152_SUS_CTP_Reg_3DT/...

Patient ID mapping:
  nnUNet case_NNNN  <->  ISLES sub-strokeNNNN
  (e.g. case_0001  <->  sub-stroke0001)

Two output resolutions (same as the SUS version):
  --resolution native  (default)
      One row per stage-3 voxel.
  --resolution ncct
      One row per NCCT foreground voxel (upsampled features).

Aggregation: voxel CSV -> slice CSV -> patient CSV (mean/median/max pooling).
"""

import os, sys, glob, json, pickle, importlib, argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import SimpleITK as sitk
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor

# =========================================================
# CONFIG
# =========================================================
DATASET_ROOT  = os.environ.get("NNUNET_DATASET_ROOT", "/path/to/nnUNetFrame/dataset")

# Model is trained on SUS — we reuse its weights/plans/dataset.json
MODEL_DATASET = "Dataset1152_SUS_CTP_Reg_3DT"
# Raw inputs come from the ISLES dataset
INPUT_DATASET = "Dataset1155_ISLES24_CTP_3DMultiChannel"

TRAINER_NAME  = "nnUNetTrainer__nnUNetPlans__3d_fullres"
CONFIGURATION = "3d_fullres"

ISLES_BASE       = os.environ.get("ISLES24_WORK_ROOT", "/path/to/isles24-work")
LABELS_DIR       = os.path.join(ISLES_BASE, "analysis")  # flat dir of <pid>_6class_map.nii.gz

RAW_TS       = os.path.join(DATASET_ROOT, "nnUNet_raw", INPUT_DATASET, "imagesTs")
MODEL_DIR    = os.path.join(DATASET_ROOT, "nnUNet_trained_models", MODEL_DATASET, TRAINER_NAME)
PLANS_PATH   = os.path.join(MODEL_DIR, "plans.json")
DATASET_JSON = os.path.join(DATASET_ROOT, "nnUNet_raw", MODEL_DATASET, "dataset.json")

OUT_DIR = os.path.join(os.environ.get("FEATURE_WORK_ROOT", "/path/to/feature-work"), "nnUnet_features_extraction", "features_3d_isles")

INPUT_CHANNELS = 40
NUM_CLASSES    = 4
ENC_STAGE      = 3
FEATURE_DIM    = 256

# Cumulative encoder strides through stages 0..3 (from plans.json)
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
# CASE LOADING
# =========================================================
_CASE_RE = re.compile(r"^case_(\d+)_\d+\.nii\.gz$")

def case_to_pid(case_id):
    """case_0001 -> sub-stroke0001"""
    num = case_id.split("case_")[-1]
    return f"sub-stroke{num}"

def pid_to_case(pid):
    """sub-stroke0001 -> case_0001"""
    num = pid.replace("sub-stroke", "")
    return f"case_{num}"

def discover_cases():
    """Find all unique case_NNNN ids in imagesTs/ that have an existing 6-class mask."""
    case_ids = set()
    for fname in os.listdir(RAW_TS):
        m = _CASE_RE.match(fname)
        if m:
            case_ids.add(f"case_{m.group(1)}")
    case_ids = sorted(case_ids)

    valid = []
    for c in case_ids:
        pid = case_to_pid(c)
        mask_p = os.path.join(LABELS_DIR, f"{pid}_6class_map.nii.gz")
        if os.path.exists(mask_p):
            valid.append(c)
    return valid


def preprocess_test_case(case_id, plans_manager, configuration_manager, dataset_json):
    pp = DefaultPreprocessor(verbose=False)
    img_paths = sorted(glob.glob(os.path.join(RAW_TS, f"{case_id}_*.nii.gz")))
    if len(img_paths) != INPUT_CHANNELS:
        raise RuntimeError(f"{case_id}: expected {INPUT_CHANNELS} channel files, found {len(img_paths)}")
    data, _, props = pp.run_case(img_paths, None, plans_manager, configuration_manager, dataset_json)
    return data.astype(np.float32), props


def load_6class_mask(pid):
    p = os.path.join(LABELS_DIR, f"{pid}_6class_map.nii.gz")
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
    mask_crop = mask_ncct[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi]

    Zp, Hp, Wp = preprocessed_shape
    if mask_crop.shape[1:] != (Hp, Wp):
        mt = torch.from_numpy(mask_crop.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        mt = F.interpolate(mt, size=mask_crop.shape[:1] + (Hp, Wp), mode='nearest')
        mask_crop = mt.squeeze(0).squeeze(0).numpy().astype(np.int32)

    if mask_crop.shape[0] != Zp:
        if mask_crop.shape[0] < Zp:
            pad = Zp - mask_crop.shape[0]
            mask_crop = np.pad(mask_crop, ((0, pad), (0, 0), (0, 0)), constant_values=0)
        else:
            mask_crop = mask_crop[:Zp]
    return mask_crop


def block_pool_mask_to_stage3(mask_p, stride=S3_STRIDES, threshold=0.3):
    Z, H, W = mask_p.shape
    sz, sh, sw = stride
    Zs, Hs, Ws = Z // sz, H // sh, W // sw
    if Zs == 0 or Hs == 0 or Ws == 0:
        return np.zeros((max(Zs, 1), max(Hs, 1), max(Ws, 1)), dtype=np.int32)
    m = mask_p[:Zs*sz, :Hs*sh, :Ws*sw]
    block_size = sz * sh * sw

    counts = np.zeros((7, Zs, Hs, Ws), dtype=np.int32)
    for cls in range(7):
        cm = (m == cls).astype(np.int32).reshape(Zs, sz, Hs, sh, Ws, sw)
        counts[cls] = cm.sum(axis=(1, 3, 5))

    fg_counts = counts[1:].sum(axis=0)
    fg_frac   = fg_counts.astype(np.float32) / block_size
    fg_arg    = counts[1:].argmax(axis=0) + 1
    out = np.where(fg_frac >= threshold, fg_arg, 0).astype(np.int32)
    return out


# =========================================================
# COORDINATE MAPPING
# =========================================================
def stage3_to_ncct_coords(z_s, h_s, w_s, props, preprocessed_shape, stride=S3_STRIDES):
    bbox = props["bbox_used_for_cropping"]
    z_lo, z_hi = bbox[0]; y_lo, y_hi = bbox[1]; x_lo, x_hi = bbox[2]
    Zp, Hp, Wp = preprocessed_shape
    H_crop = y_hi - y_lo
    W_crop = x_hi - x_lo
    sz, sh, sw = stride

    z_p_center = sz * z_s + sz // 2
    h_p_center = sh * h_s + sh // 2
    w_p_center = sw * w_s + sw // 2

    z_ncct = z_lo + z_p_center
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
    C, Z, H, W = data.shape
    pZ, pH, pW = patch_size
    sz, sh, sw = stride
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
    C, Z, H, W = data.shape
    pZ, pH, pW = patch_size
    sz, sh, sw = S3_STRIDES

    data_padded, (Zp, Hp, Wp) = _pad_for_native(data, patch_size, S3_STRIDES)
    pZ_s, pH_s, pW_s = pZ // sz, pH // sh, pW // sw

    if mode == 'native':
        Zs, Hs, Ws = Zp // sz, Hp // sh, Wp // sw
        feat_sum = torch.zeros((FEATURE_DIM, Zs, Hs, Ws), dtype=torch.float32, device=device)
        weight   = torch.zeros((1, Zs, Hs, Ws),           dtype=torch.float32, device=device)
    else:
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
                    f = hook_cache["feat"]
                    f_acc = f if f_acc is None else f_acc + f
                f_mean = (f_acc / len(fold_state_dicts)).squeeze(0)

                if mode == 'native':
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
    if mode == 'native':
        Zs_orig = (Z + sz - 1) // sz
        Hs_orig = (H + sh - 1) // sh
        Ws_orig = (W + sw - 1) // sw
        feat_avg = feat_avg[:, :Zs_orig, :Hs_orig, :Ws_orig].contiguous().cpu().numpy()
    else:
        feat_avg = feat_avg[:, :Z, :H, :W].contiguous().cpu().numpy()
    return feat_avg


# =========================================================
# NCCT MODE: per-slice inverse + voxel rows
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
# PER-CASE: write voxel rows
# =========================================================
def process_case(case_id, model, fold_sds, hook_cache,
                 plans_manager, config_manager, dataset_json,
                 patch_size, step_size, device, resolution,
                 out_handle, header_written):
    pid = case_to_pid(case_id)

    print(f"  preprocessing {case_id} ({pid}) ...", flush=True)
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
        mask_p   = resample_mask_to_preprocessed(mask_ncct, props, img.shape[1:])
        mask_s3  = block_pool_mask_to_stage3(mask_p, stride=S3_STRIDES, threshold=0.3)
        Zs, Hs, Ws = feat_avg.shape[1:]
        m = mask_s3[:Zs, :Hs, :Ws]
        if m.shape != (Zs, Hs, Ws):
            pad = [(0, max(0, t - s)) for s, t in zip(m.shape, (Zs, Hs, Ws))]
            m = np.pad(m, pad, constant_values=0)
        zsa, hsa, wsa = np.where((m >= 1) & (m <= 6))
        if len(zsa) > 0:
            labels = m[zsa, hsa, wsa]
            feats  = feat_avg[:, zsa, hsa, wsa].T
            z_ncct, h_ncct, w_ncct = stage3_to_ncct_coords(
                zsa, hsa, wsa, props, img.shape[1:], stride=S3_STRIDES
            )
            meta = pd.DataFrame({
                "patient_id": pid,
                "case_id":    case_id,
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
                "case_id":    case_id,
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
    print(f"\nAggregating from {voxel_csv} ...", flush=True)
    df = pd.read_csv(voxel_csv)
    print(f"  total rows: {len(df):,}", flush=True)

    z_key = "z_ncct" if "z_ncct" in df.columns else "z"

    # Slice-level
    slice_rows = []
    group_cols = ["patient_id", z_key, "label", "class_name"]
    for keys, g in df.groupby(group_cols):
        rec = dict(zip(group_cols, keys))
        rec["n_voxels"] = len(g)
        v = g[FEAT_COLS].values
        for stat_name, stat_func in [("mean", np.mean), ("median", np.median), ("max", np.max)]:
            vec = stat_func(v, axis=0)
            for j, c in enumerate(FEAT_COLS):
                rec[f"{c}_{stat_name}"] = vec[j]
        slice_rows.append(rec)
    slice_df = pd.DataFrame(slice_rows)
    slice_df.to_csv(slice_csv, index=False, float_format="%.5f")
    print(f"  wrote slice-level CSV: {slice_csv}  ({len(slice_df):,} rows)", flush=True)

    # Patient-level (pool over slice means)
    patient_rows = []
    base_cols = [f"{c}_mean" for c in FEAT_COLS]
    pgroup_cols = ["patient_id", "label", "class_name"]
    for keys, g in slice_df.groupby(pgroup_cols):
        rec = dict(zip(pgroup_cols, keys))
        rec["n_slices"] = len(g)
        rec["n_voxels_total"] = int(g["n_voxels"].sum())
        v = g[base_cols].values
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
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")
    print(f"Input  dataset: {INPUT_DATASET}")
    print(f"Model  dataset: {MODEL_DATASET}")
    print(f"Resolution mode: {args.resolution}")
    torch.backends.cudnn.benchmark = True

    voxel_csv   = os.path.join(OUT_DIR, f"voxel_features_isles_{args.resolution}.csv")
    slice_csv   = os.path.join(OUT_DIR, f"slice_features_isles_{args.resolution}.csv")
    patient_csv = os.path.join(OUT_DIR, f"patient_features_isles_{args.resolution}.csv")
    print(f"Output:\n  {voxel_csv}\n  {slice_csv}\n  {patient_csv}")

    with open(PLANS_PATH) as f:   plans = json.load(f)
    with open(DATASET_JSON) as f: dataset_json = json.load(f)
    cfg = plans["configurations"][CONFIGURATION]
    PATCH_SIZE = tuple(cfg["patch_size"])
    STEP = tuple(max(1, p // 2) for p in PATCH_SIZE)
    print(f"patch_size={PATCH_SIZE}  step={STEP}  S3_strides={S3_STRIDES}")

    model, arch_kwargs, _ = build_model(plans, device)
    fold_sds = load_fold_checkpoints()
    print(f"Loaded {len(fold_sds)} fold checkpoints from {MODEL_DATASET}")

    hook_cache = {}
    def _enc_hook(module, inp, out):
        hook_cache["feat"] = out.detach()
    model.encoder.stages[ENC_STAGE].register_forward_hook(_enc_hook)

    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    plans_manager  = PlansManager(plans)
    config_manager = plans_manager.get_configuration(CONFIGURATION)

    all_cases = discover_cases()
    end = args.end if args.end is not None else len(all_cases)
    cases = all_cases[args.start:end]
    print(f"Processing {len(cases)} ISLES cases (start={args.start}, end={end}) "
          f"out of {len(all_cases)} cases with matching masks")

    if not args.skip_extract:
        mode_open = "a" if args.start > 0 and os.path.exists(voxel_csv) else "w"
        out_handle = open(voxel_csv, mode_open, newline="")
        header_written = {"done": (mode_open == "a")}
        try:
            for i, case_id in enumerate(cases, 1):
                print(f"\n[{i}/{len(cases)}] {case_id} ({case_to_pid(case_id)})", flush=True)
                try:
                    process_case(case_id, model, fold_sds, hook_cache,
                                 plans_manager, config_manager, dataset_json,
                                 PATCH_SIZE, STEP, device, args.resolution,
                                 out_handle, header_written)
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
