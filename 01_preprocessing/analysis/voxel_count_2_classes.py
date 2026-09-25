"""
voxel_count_2_classes.py
========================
Compute per-slice and per-patient voxel / volume statistics for
the 3-class (background / penumbra / core) CTP lesion masks.

Only slices where a sufficient fraction of the label voxels fall inside
CTP signal coverage are kept (controlled by ``--min-overlap``).

Standalone usage
----------------
    python voxel_count_2_classes.py
    python voxel_count_2_classes.py --root-dir /data/derivatives \\
        --out-slice-csv slices.csv --out-patient-csv patients.csv

Pipeline usage
--------------
    from analysis.voxel_count_2_classes import extract_3class_stats
"""

import os
import glob
import argparse

import numpy as np
import pandas as pd
import SimpleITK as sitk

try:
    from config import REG_ROOT
except ImportError:
    REG_ROOT = os.environ.get("SUS_WORK_ROOT", "/path/to/sus-work")

# =========================================================
# DEFAULT CONFIG
# =========================================================
_DEFAULT_ROOT_DIR = os.path.join(os.environ.get("SUS_CURATED_ROOT", "/path/to/Curated-SUS-2025"), "derivatives")
_DEFAULT_OUT_SLICE_CSV = os.path.join(
    REG_ROOT, "voxel_counts", "3class_core_pen_slice_volume_stats_ctpvalid.csv"
)
_DEFAULT_OUT_PATIENT_CSV = os.path.join(
    REG_ROOT, "voxel_counts", "3class_core_pen_patient_volume_stats_ctpvalid.csv"
)

# 3-class label map
# background=0, penumbra=2, core=3
LABEL_MAP = {
    2: "penumbra",
    3: "core",
}

# Keep only slices where enough of the label is inside CTP coverage
MIN_OVERLAP_RATIO = 0.5

# If True, skip slices where that label has 0 voxels
KEEP_ONLY_NONZERO_LABEL_SLICES = True


# =========================================================
# HELPERS
# =========================================================
def get_patient_id_and_session(path):
    parts = path.split(os.sep)
    patient_id = next((p for p in parts if p.startswith("sub-stroke_")), None)
    session = next((p for p in parts if p.startswith("ses-")), "NA")
    return patient_id, session


def find_ctp_for_mask(mask_path, ctp_glob_name="*space-ncct_ctp.nii.gz"):
    ses_dir = os.path.dirname(mask_path)
    candidates = sorted(glob.glob(os.path.join(ses_dir, ctp_glob_name)))

    if len(candidates) == 0:
        raise FileNotFoundError(f"No CTP found in {ses_dir} with pattern {ctp_glob_name}")

    if len(candidates) > 1:
        print(f"Warning: multiple CTP files found in {ses_dir}, using first one:\n{candidates[0]}")

    return candidates[0]


def get_ctp_signal_mask(ctp_path):
    """
    Returns boolean CTP signal mask with shape (Z, Y, X)

    SimpleITK -> numpy:
      3D image: (z, y, x)
      4D image: (t, z, y, x)
    """
    ctp_img = sitk.ReadImage(ctp_path)
    ctp_arr = sitk.GetArrayFromImage(ctp_img)

    if ctp_arr.ndim == 3:
        signal_mask = (ctp_arr != 0)

    elif ctp_arr.ndim == 4:
        signal_mask = np.any(ctp_arr != 0, axis=0)

    else:
        raise ValueError(f"Unexpected CTP ndim={ctp_arr.ndim} for file: {ctp_path}")

    return signal_mask.astype(bool)


def get_valid_slices_for_label(ctp_signal_mask, label_mask_3d, min_overlap_ratio=0.5):
    """
    A slice is valid for a given label if:
      overlap(label, ctp_signal) / label_voxels >= min_overlap_ratio
    """
    valid_slices = np.zeros(label_mask_3d.shape[0], dtype=bool)
    overlap_ratios = np.full(label_mask_3d.shape[0], np.nan, dtype=float)

    for z in range(label_mask_3d.shape[0]):
        label_slice = label_mask_3d[z] > 0
        n_labelled = int(label_slice.sum())

        if n_labelled == 0:
            continue

        ctp_slice = ctp_signal_mask[z]
        overlap = int((label_slice & ctp_slice).sum())
        overlap_ratio = overlap / n_labelled
        overlap_ratios[z] = overlap_ratio

        if overlap_ratio >= min_overlap_ratio:
            valid_slices[z] = True

    return valid_slices, overlap_ratios


def extract_3class_stats(mask_path, label_map, min_overlap_ratio=0.5):
    # Read mask
    mask_img = sitk.ReadImage(mask_path)
    mask_arr = sitk.GetArrayFromImage(mask_img)   # (z, y, x)
    sx, sy, sz = mask_img.GetSpacing()            # (x, y, z)

    if mask_arr.ndim != 3:
        raise ValueError(f"3-class mask must be 3D, got shape {mask_arr.shape} for {mask_path}")

    voxel_volume_mm3 = sx * sy * sz
    patient_id, session = get_patient_id_and_session(mask_path)

    # Find matching CTP and signal mask
    ctp_path = find_ctp_for_mask(mask_path)
    ctp_signal_mask = get_ctp_signal_mask(ctp_path)

    if ctp_signal_mask.shape != mask_arr.shape:
        raise ValueError(
            f"Shape mismatch between 3-class mask and CTP signal mask:\n"
            f"mask shape = {mask_arr.shape} ({mask_path})\n"
            f"ctp  shape = {ctp_signal_mask.shape} ({ctp_path})"
        )

    slice_rows = []
    patient_rows = []

    for label_value, label_name in label_map.items():
        label_mask_3d = (mask_arr == label_value).astype(np.uint8)

        valid_slices, overlap_ratios = get_valid_slices_for_label(
            ctp_signal_mask=ctp_signal_mask,
            label_mask_3d=label_mask_3d,
            min_overlap_ratio=min_overlap_ratio,
        )

        total_voxels_3d = int(label_mask_3d[valid_slices].sum())
        total_volume_ml = (total_voxels_3d * voxel_volume_mm3) / 1000.0

        patient_rows.append({
            "patient_id": patient_id,
            "session": session,
            "label_value": label_value,
            "label_name": label_name,
            "total_voxels_3d": total_voxels_3d,
            "total_volume_ml": total_volume_ml,
            "n_mask_slices": mask_arr.shape[0],
            "n_valid_slices_for_label": int(valid_slices.sum()),
            "min_overlap_ratio": min_overlap_ratio,
        })

        for z in range(label_mask_3d.shape[0]):
            slice_voxels = int(label_mask_3d[z].sum())

            if KEEP_ONLY_NONZERO_LABEL_SLICES and slice_voxels == 0:
                continue

            is_valid = bool(valid_slices[z])
            overlap_ratio = overlap_ratios[z]

            if not is_valid:
                continue

            slice_volume_ml = (slice_voxels * voxel_volume_mm3) / 1000.0

            slice_rows.append({
                "patient_id": patient_id,
                # "session": session,
                "slice_idx": z,
                "label_value": label_value,
                "label_name": label_name,
                "slice_voxels": slice_voxels,
                "slice_volume_ml": slice_volume_ml,
                # "overlap_ratio": overlap_ratio,
                "slice_valid_for_label": is_valid,
                "total_voxels_3d": total_voxels_3d,
                "total_volume_ml": total_volume_ml,
                "n_mask_slices": mask_arr.shape[0],
                "n_valid_slices_for_label": int(valid_slices.sum()),
                "min_overlap_ratio": min_overlap_ratio,
            })

    return slice_rows, patient_rows


# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Compute voxel/volume stats for 3-class CTP lesion masks."
    )
    parser.add_argument("--root-dir", type=str, default=_DEFAULT_ROOT_DIR,
                        help="Root derivatives directory")
    parser.add_argument("--out-slice-csv", type=str, default=_DEFAULT_OUT_SLICE_CSV,
                        help="Output CSV path for slice-level stats")
    parser.add_argument("--out-patient-csv", type=str, default=_DEFAULT_OUT_PATIENT_CSV,
                        help="Output CSV path for patient-level stats")
    parser.add_argument("--min-overlap", type=float, default=MIN_OVERLAP_RATIO,
                        help="Minimum CTP overlap ratio to consider a slice valid")
    args = parser.parse_args()

    mask_glob = os.path.join(
        args.root_dir, "sub-stroke_*", "ses-*",
        "*space-ncct_ctp_lesion_msk_3class.nii.gz",
    )

    mask_files = sorted(glob.glob(mask_glob))
    print(f"Found {len(mask_files)} 3-class lesion mask files")

    all_slice_rows = []
    all_patient_rows = []

    for mask_path in mask_files:
        try:
            slice_rows, patient_rows = extract_3class_stats(
                mask_path,
                LABEL_MAP,
                min_overlap_ratio=args.min_overlap,
            )
            all_slice_rows.extend(slice_rows)
            all_patient_rows.extend(patient_rows)
            print(f"Processed: {mask_path}")
        except Exception as e:
            print(f"Error processing {mask_path}: {e}")

    # ---------------- Slice-level CSV ----------------
    df_slice = pd.DataFrame(all_slice_rows)
    slice_col_order = [
        "patient_id",
        # "session",
        "slice_idx",
        "label_value",
        "label_name",
        "slice_voxels",
        "slice_volume_ml",
        # "overlap_ratio",
        "slice_valid_for_label",
        "total_voxels_3d",
        "total_volume_ml",
        "n_mask_slices",
        "n_valid_slices_for_label",
        "min_overlap_ratio",
    ]
    df_slice = df_slice[slice_col_order]
    os.makedirs(os.path.dirname(args.out_slice_csv), exist_ok=True)
    df_slice.to_csv(args.out_slice_csv, index=False)

    # ---------------- Patient-level CSV ----------------
    df_patient = pd.DataFrame(all_patient_rows)
    patient_col_order = [
        "patient_id",
        "session",
        "label_value",
        "label_name",
        "total_voxels_3d",
        "total_volume_ml",
        "n_mask_slices",
        "n_valid_slices_for_label",
        "min_overlap_ratio",
    ]
    df_patient = df_patient[patient_col_order]
    df_patient = df_patient.drop_duplicates(
        subset=["patient_id", "session", "label_value", "label_name"]
    ).reset_index(drop=True)

    df_patient.to_csv(args.out_patient_csv, index=False)

    print(f"\nSaved slice-level CSV to:\n{args.out_slice_csv}")
    print(f"Saved patient-level CSV to:\n{args.out_patient_csv}")

    print("\nSlice-level preview:")
    print(df_slice.head())

    print("\nPatient-level preview:")
    print(df_patient.head())


if __name__ == "__main__":
    main()
