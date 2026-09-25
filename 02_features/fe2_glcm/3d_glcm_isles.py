#!/usr/bin/env python3
"""3d_glcm_isles.py

Multi-patient 4D CTP -> 3D GLCM feature extraction for ISLES24.

Mirrors sus25_feat_extraction/3d_glcm/3d_glcm.py but adapted to ISLES:
  - Patient IDs are 'sub-stroke<NNNN>' (single segment, no underscores).
  - CTP volumes live under
      <derivatives>/<pid>/ses-01/<pid>_ses-01_space-ncct_ctp.nii.gz
  - 6-class labels live in a FLAT folder:
      <labels_dir>/<pid>_6class_map.nii.gz
    (this is ISLES24/analysis/, the union of 5_Class + 6_Class — cases
    without CLB simply contain no label-5 voxels)

Label semantics match SUS:
  1=core_fi, 2=core_brain, 3=pen_fi, 4=pen_brain, 5=CLB_brain, 6=NHB_fi
"""
import os

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import SimpleITK as sitk

try:
    from radiomics.glcm import RadiomicsGLCM
except Exception as e:
    raise SystemExit(
        "Install PyRadiomics first:\n"
        "conda install -c conda-forge pyradiomics simpleitk\n"
        f"Original error: {e}"
    )


# -------------------------------------------------------
# Extract patient ID from folder / filename
# -------------------------------------------------------
def extract_pid(name):
    """
    From folder name or filename:
    sub-stroke0001  ->  sub-stroke0001
    """
    m = re.match(r"(sub-stroke\d+)", name)
    return m.group(1) if m else None


# -------------------------------------------------------
# GLCM computation
# -------------------------------------------------------
def compute_glcm_features(image_hwt, mask_hw, settings):

    mask_3d = np.repeat(mask_hw[:, :, None], image_hwt.shape[2], axis=2)

    img_sitk = sitk.GetImageFromArray(image_hwt.astype(np.float32))
    mask_sitk = sitk.GetImageFromArray(mask_3d.astype(np.uint8))

    glcm = RadiomicsGLCM(img_sitk, mask_sitk, label=settings["label"])
    glcm.enableAllFeatures()
    glcm.settings.update({
        "binWidth": settings["binWidth"],
        "distances": settings["distances"],
        "force2D": settings["force2D"],
        "force2Ddimension": settings["force2Ddimension"],
        "symmetricalGLCM": settings["symmetricalGLCM"],
    })

    feats_raw = glcm.execute()

    feats = {}
    for k, v in feats_raw.items():
        if k.startswith("diagnostics_"):
            continue
        try:
            feats[k] = float(np.nanmean(np.asarray(v)))
        except Exception:
            continue

    return feats


# -------------------------------------------------------
# Process one patient
# -------------------------------------------------------
def process_patient(pid, ctp_path, label_path, settings):

    ctp_np = sitk.GetArrayFromImage(sitk.ReadImage(str(ctp_path)))      # (T,Z,Y,X)
    label_np = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path)))  # (Z,Y,X)

    # Convert to (Z,Y,X,T)
    ctp_np = np.transpose(ctp_np, (1, 2, 3, 0))

    Z, H, W, T = ctp_np.shape

    if label_np.shape[0] != Z:
        raise ValueError(f"{pid}: Z mismatch between CTP and label")

    label_map = {
        1: "core_fi",
        2: "core_brain",
        3: "pen_fi",
        4: "pen_brain",
        5: "CLB_brain",
        6: "NHB_fi"
    }

    rows = []

    for z in range(Z):

        img3d = ctp_np[z]
        label_slice = label_np[z]

        if np.sum(img3d) == 0:
            continue

        if not np.any((label_slice == 3) | (label_slice == 4)):
            continue

        for lbl_val, region_name in label_map.items():

            valid_ctp_mask = np.sum(img3d, axis=2) > 0
            mask = (label_slice == lbl_val)
            mask = mask & valid_ctp_mask
            mask = mask.astype(np.uint8)

            if mask.sum() == 0:
                continue

            feats = compute_glcm_features(img3d, mask, settings)
            if not feats:
                continue

            rec = {
                "patient_id": pid,
                "slice": z,
                "region": region_name
            }
            rec.update(feats)
            rows.append(rec)

    return pd.DataFrame(rows)


# -------------------------------------------------------
# CLI
# -------------------------------------------------------
def parse_list(s):
    return [x.strip() for x in s.split(",") if x.strip() != ""]

def parse_distances(s):
    return [int(x) for x in s.split(",") if x.strip() != ""]


def main():

    ISLES_BASE = Path(os.environ.get("ISLES24_WORK_ROOT", "/path/to/isles24-work"))
    DEFAULT_DERIV  = ISLES_BASE / "isles24_curated" / "derivatives"
    DEFAULT_LABELS = ISLES_BASE / "analysis"

    ap = argparse.ArgumentParser(
        description="Multi-patient 4D CTP -> 3D GLCM extraction (ISLES24)"
    )

    ap.add_argument("--derivatives-dir", type=Path, default=DEFAULT_DERIV,
                    help="Root derivatives folder (contains sub-stroke<NNNN>/)")
    ap.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS,
                    help="Flat folder with <pid>_6class_map.nii.gz files")

    ap.add_argument("--patients", type=str,
                    help="Comma-separated patient IDs to run")

    ap.add_argument("--start-idx", type=int)
    ap.add_argument("--end-idx", type=int)

    ap.add_argument("--bin-width", type=float, default=8.0)
    ap.add_argument("--distances", type=parse_distances, default=[1])
    ap.add_argument("--force-2d", action="store_true")
    ap.add_argument("--force-2d-dim", type=int, default=0)
    ap.add_argument("--no-symmetric-glcm", action="store_true")
    ap.add_argument("--label", type=int, default=1)

    ap.add_argument("--out-csv", type=Path,
                    default=Path("glcm_isles24.csv"))

    args = ap.parse_args()

    settings = dict(
        binWidth=args.bin_width,
        distances=args.distances,
        force2D=args.force_2d,
        force2Ddimension=args.force_2d_dim,
        symmetricalGLCM=(not args.no_symmetric_glcm),
        label=args.label,
    )

    # -------------------------------------------------------
    # Discover patients from derivatives/sub-stroke<NNNN>/ses-01/
    # Labels are looked up in the flat --labels-dir.
    # -------------------------------------------------------
    patient_map = {}   # pid -> {"ctp": Path, "label": Path}

    for subdir in sorted(args.derivatives_dir.iterdir()):
        if not subdir.is_dir():
            continue
        pid = extract_pid(subdir.name)
        if pid is None:
            continue

        ses_dir = subdir / "ses-01"
        if not ses_dir.is_dir():
            continue

        ctp_path   = ses_dir / f"{pid}_ses-01_space-ncct_ctp.nii.gz"
        label_path = args.labels_dir / f"{pid}_6class_map.nii.gz"

        if ctp_path.exists():
            patient_map[pid] = {"ctp": ctp_path, "label": label_path}

    all_patients = sorted(patient_map.keys())

    if args.patients:
        selected = parse_list(args.patients)
    elif args.start_idx and args.end_idx:
        selected = all_patients[args.start_idx-1:args.end_idx]
    else:
        selected = all_patients

    print(f"Total CTP patients found: {len(all_patients)}")
    print(f"Selected: {len(selected)}")

    all_rows = []

    for pid in selected:

        print(f"\n=== Processing {pid} ===")

        if pid not in patient_map:
            print("  Missing CTP")
            continue

        ctp_path = patient_map[pid]["ctp"]
        label_path = patient_map[pid]["label"]

        if not label_path.exists():
            print(f"  Missing 6-class label: {label_path}")
            continue

        try:
            df = process_patient(pid, ctp_path, label_path, settings)

            if df.empty:
                print("  (no valid slices)")
            else:
                print(f"  -> {len(df)} rows")
                all_rows.append(df)

        except Exception as e:
            print(f"❌ Error: {e}")

    if not all_rows:
        print("\n⚠ No data extracted.")
        return

    final_df = pd.concat(all_rows, ignore_index=True)
    final_df.to_csv(args.out_csv, index=False)

    print(f"\n✅ Saved: {args.out_csv}")
    print(f"Total rows: {len(final_df)}")


if __name__ == "__main__":
    main()
