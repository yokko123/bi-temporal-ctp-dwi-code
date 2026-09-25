"""
voxel_count_dwi.py
==================
Compute per-slice and per-patient voxel / volume statistics for
DWI final-infarct lesion masks (binary, label=1).

Only slices where a sufficient fraction of the lesion voxels fall inside
DWI signal coverage are kept (controlled by ``--min-overlap``).

Standalone usage
----------------
    python voxel_count_dwi.py
    python voxel_count_dwi.py --root-dir /data/derivatives \\
        --out-slice-csv slices.csv --out-patient-csv patients.csv

Pipeline usage
--------------
    from analysis.voxel_count_dwi import extract_dwi_lesion_stats
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
    REG_ROOT, "voxel_counts", "dwi_final_infarct_slice_volume_stats_valid.csv"
)
_DEFAULT_OUT_PATIENT_CSV = os.path.join(
    REG_ROOT, "voxel_counts", "dwi_final_infarct_patient_volume_stats_valid.csv"
)

# DWI lesion mask in ses-02
DWI_GLOB_NAME = "*space-ncct_dwi.nii.gz"

# Binary final infarct label
LABEL_VALUE = 1
LABEL_NAME = "final_infarct"

# Keep only slices where enough of the lesion is inside DWI coverage
MIN_OVERLAP_RATIO = 0.5

# If True, skip slices where lesion has 0 voxels
KEEP_ONLY_NONZERO_LABEL_SLICES = True


# =========================================================
# HELPERS
# =========================================================
def get_patient_id_and_session(path):
    parts = path.split(os.sep)
    patient_id = next((p for p in parts if p.startswith("sub-stroke_")), None)
    session = next((p for p in parts if p.startswith("ses-")), "NA")
    return patient_id, session


def find_dwi_for_mask(mask_path):
    ses_dir = os.path.dirname(mask_path)
    candidates = sorted(glob.glob(os.path.join(ses_dir, DWI_GLOB_NAME)))

    if len(candidates) == 0:
        raise FileNotFoundError(f"No DWI found in {ses_dir} with pattern {DWI_GLOB_NAME}")

    if len(candidates) > 1:
        print(f"Warning: multiple DWI files found in {ses_dir}, using first one:\n{candidates[0]}")

    return candidates[0]


def get_dwi_signal_mask(dwi_path):
    """
    Returns boolean DWI signal mask with shape (Z, Y, X)

    SimpleITK -> numpy:
      3D image: (z, y, x)
      4D image: (t, z, y, x)
    """
    dwi_img = sitk.ReadImage(dwi_path)
    dwi_arr = sitk.GetArrayFromImage(dwi_img)

    if dwi_arr.ndim == 3:
        signal_mask = (dwi_arr != 0)

    elif dwi_arr.ndim == 4:
        signal_mask = np.any(dwi_arr != 0, axis=0)

    else:
        raise ValueError(f"Unexpected DWI ndim={dwi_arr.ndim} for file: {dwi_path}")

    return signal_mask.astype(bool)


def get_valid_slices_for_label(signal_mask_3d, label_mask_3d, min_overlap_ratio=0.5):
    """
    A slice is valid if:
      overlap(label, signal) / label_voxels >= min_overlap_ratio
    """
    valid_slices = np.zeros(label_mask_3d.shape[0], dtype=bool)
    overlap_ratios = np.full(label_mask_3d.shape[0], np.nan, dtype=float)

    for z in range(label_mask_3d.shape[0]):
        label_slice = label_mask_3d[z] > 0
        n_labelled = int(label_slice.sum())

        if n_labelled == 0:
            continue

        signal_slice = signal_mask_3d[z]
        overlap = int((label_slice & signal_slice).sum())
        overlap_ratio = overlap / n_labelled
        overlap_ratios[z] = overlap_ratio

        if overlap_ratio >= min_overlap_ratio:
            valid_slices[z] = True

    return valid_slices, overlap_ratios


def extract_dwi_lesion_stats(mask_path, min_overlap_ratio=0.5):
    # Read lesion mask
    mask_img = sitk.ReadImage(mask_path)
    mask_arr = sitk.GetArrayFromImage(mask_img)   # (z, y, x)
    sx, sy, sz = mask_img.GetSpacing()            # (x, y, z)

    if mask_arr.ndim != 3:
        raise ValueError(f"DWI lesion mask must be 3D, got shape {mask_arr.shape} for {mask_path}")

    voxel_volume_mm3 = sx * sy * sz
    patient_id, session = get_patient_id_and_session(mask_path)

    # Read matching DWI and make signal mask
    dwi_path = find_dwi_for_mask(mask_path)
    dwi_signal_mask = get_dwi_signal_mask(dwi_path)

    if dwi_signal_mask.shape != mask_arr.shape:
        raise ValueError(
            f"Shape mismatch between DWI lesion mask and DWI signal mask:\n"
            f"mask shape = {mask_arr.shape} ({mask_path})\n"
            f"dwi  shape = {dwi_signal_mask.shape} ({dwi_path})"
        )

    # Binary lesion
    label_mask_3d = (mask_arr == LABEL_VALUE).astype(np.uint8)

    valid_slices, overlap_ratios = get_valid_slices_for_label(
        signal_mask_3d=dwi_signal_mask,
        label_mask_3d=label_mask_3d,
        min_overlap_ratio=min_overlap_ratio,
    )

    total_voxels_3d = int(label_mask_3d[valid_slices].sum())
    total_volume_ml = (total_voxels_3d * voxel_volume_mm3) / 1000.0

    patient_row = {
        "patient_id": patient_id,
        "session": session,
        "label_value": LABEL_VALUE,
        "label_name": LABEL_NAME,
        "total_voxels_3d": total_voxels_3d,
        "total_volume_ml": total_volume_ml,
        "n_mask_slices": mask_arr.shape[0],
        "n_valid_slices_for_label": int(valid_slices.sum()),
        "min_overlap_ratio": min_overlap_ratio,
    }

    slice_rows = []
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
            "session": session,
            "slice_idx": z,
            "label_value": LABEL_VALUE,
            "label_name": LABEL_NAME,
            "slice_voxels": slice_voxels,
            "slice_volume_ml": slice_volume_ml,
            # "overlap_ratio": overlap_ratio,
            "slice_valid_for_label": is_valid,
            "total_voxels_3d": total_voxels_3d,
            "total_volume_ml": total_volume_ml,
            "n_mask_slices": mask_arr.shape[0],
            "n_valid_slices_for_label": int(valid_slices.sum()),
            # "min_overlap_ratio": min_overlap_ratio,
        })

    return slice_rows, patient_row


# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Compute voxel/volume stats for DWI final-infarct lesion masks."
    )
    parser.add_argument("--root-dir", type=str, default=_DEFAULT_ROOT_DIR,
                        help="Root derivatives directory")
    parser.add_argument("--out-slice-csv", type=str, default=_DEFAULT_OUT_SLICE_CSV,
                        help="Output CSV path for slice-level stats")
    parser.add_argument("--out-patient-csv", type=str, default=_DEFAULT_OUT_PATIENT_CSV,
                        help="Output CSV path for patient-level stats")
    parser.add_argument("--min-overlap", type=float, default=MIN_OVERLAP_RATIO,
                        help="Minimum DWI overlap ratio to consider a slice valid")
    args = parser.parse_args()

    mask_glob = os.path.join(
        args.root_dir, "sub-stroke_*", "ses-02",
        "*space-ncct_dwi_lesion_msk.nii.gz",
    )

    mask_files = sorted(glob.glob(mask_glob))
    print(f"Found {len(mask_files)} DWI lesion mask files")

    all_slice_rows = []
    all_patient_rows = []

    for mask_path in mask_files:
        try:
            slice_rows, patient_row = extract_dwi_lesion_stats(
                mask_path,
                min_overlap_ratio=args.min_overlap,
            )
            all_slice_rows.extend(slice_rows)
            all_patient_rows.append(patient_row)
            print(f"Processed: {mask_path}")
        except Exception as e:
            print(f"Error processing {mask_path}: {e}")

    # ---------------- Slice-level CSV ----------------
    df_slice = pd.DataFrame(all_slice_rows)
    slice_col_order = [
        "patient_id",
        "session",
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
        # "min_overlap_ratio",
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
