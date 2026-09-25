"""
cc_dwi.py
=========
Compute 3-D connected-component statistics for DWI final-infarct lesion masks
(binary, label=1), filtering out slices with insufficient DWI signal overlap.

Standalone usage
----------------
    python cc_dwi.py
    python cc_dwi.py --root-dir /data/derivatives --out-csv components.csv

Pipeline usage
--------------
    from analysis.cc_dwi import get_valid_slices_for_label
"""

import os
import glob
import argparse

import numpy as np
import pandas as pd
import SimpleITK as sitk
import cc3d

try:
    from config import REG_ROOT
except ImportError:
    REG_ROOT = os.environ.get("SUS_WORK_ROOT", "/path/to/sus-work")

# =========================================================
# DEFAULT CONFIG
# =========================================================
_DEFAULT_ROOT_DIR = os.path.join(os.environ.get("SUS_CURATED_ROOT", "/path/to/Curated-SUS-2025"), "derivatives")
_DEFAULT_OUT_CSV = os.path.join(
    REG_ROOT, "voxel_counts", "dwi_final_infarct_connected_components.csv"
)

DWI_GLOB_NAME = "*space-ncct_dwi.nii.gz"

LABEL_VALUE = 1
LABEL_NAME = "final_infarct"
MIN_OVERLAP_RATIO = 0.5


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
        raise FileNotFoundError(f"No DWI found in {ses_dir}")
    return candidates[0]

def get_dwi_signal_mask(dwi_path):
    dwi_img = sitk.ReadImage(dwi_path)
    dwi_arr = sitk.GetArrayFromImage(dwi_img)

    if dwi_arr.ndim == 3:
        signal_mask = (dwi_arr != 0)
    elif dwi_arr.ndim == 4:
        signal_mask = np.any(dwi_arr != 0, axis=0)
    else:
        raise ValueError(f"Unexpected DWI ndim={dwi_arr.ndim}")

    return signal_mask.astype(bool)

def get_valid_slices_for_label(signal_mask_3d, label_mask_3d, min_overlap_ratio=0.5):
    valid_slices = np.zeros(label_mask_3d.shape[0], dtype=bool)

    for z in range(label_mask_3d.shape[0]):
        label_slice = label_mask_3d[z] > 0
        n_labelled = int(label_slice.sum())
        if n_labelled == 0:
            continue

        overlap = int((label_slice & signal_mask_3d[z]).sum())
        overlap_ratio = overlap / n_labelled

        if overlap_ratio >= min_overlap_ratio:
            valid_slices[z] = True

    return valid_slices


# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Compute connected-component stats for DWI final-infarct lesion masks."
    )
    parser.add_argument("--root-dir", type=str, default=_DEFAULT_ROOT_DIR,
                        help="Root derivatives directory")
    parser.add_argument("--out-csv", type=str, default=_DEFAULT_OUT_CSV,
                        help="Output CSV path")
    parser.add_argument("--min-overlap", type=float, default=MIN_OVERLAP_RATIO,
                        help="Minimum DWI overlap ratio to consider a slice valid")
    args = parser.parse_args()

    mask_glob = os.path.join(
        args.root_dir, "sub-stroke_*", "ses-02",
        "*space-ncct_dwi_lesion_msk.nii.gz",
    )

    rows = []

    mask_files = sorted(glob.glob(mask_glob))
    print(f"Found {len(mask_files)} DWI lesion masks")

    for mask_path in mask_files:
        try:
            patient_id, session = get_patient_id_and_session(mask_path)

            mask_img = sitk.ReadImage(mask_path)
            mask_arr = sitk.GetArrayFromImage(mask_img)  # (z, y, x)
            sx, sy, sz = mask_img.GetSpacing()
            voxel_volume_ml = (sx * sy * sz) / 1000.0

            dwi_path = find_dwi_for_mask(mask_path)
            dwi_signal_mask = get_dwi_signal_mask(dwi_path)

            if dwi_signal_mask.shape != mask_arr.shape:
                raise ValueError(
                    f"Shape mismatch: mask {mask_arr.shape} vs dwi {dwi_signal_mask.shape}"
                )

            lesion_mask = (mask_arr == LABEL_VALUE).astype(np.uint8)

            valid_slices = get_valid_slices_for_label(
                dwi_signal_mask,
                lesion_mask,
                min_overlap_ratio=args.min_overlap,
            )

            filtered_mask = np.zeros_like(lesion_mask, dtype=np.uint8)
            filtered_mask[valid_slices] = lesion_mask[valid_slices]

            total_voxels = int(filtered_mask.sum())
            total_volume_ml = total_voxels * voxel_volume_ml

            if total_voxels == 0:
                n_components = 0
                component_sizes = []
            else:
                cc = cc3d.connected_components(filtered_mask, connectivity=26)
                component_ids, counts = np.unique(cc[cc > 0], return_counts=True)
                n_components = len(component_ids)
                component_sizes = counts.tolist()

            rows.append({
                "patient_id": patient_id,
                # "session": session,
                "label_value": LABEL_VALUE,
                "label_name": LABEL_NAME,
                "n_connected_components": n_components,
                "total_voxels": total_voxels,
                "total_volume_ml": total_volume_ml,
                "largest_component_voxels": max(component_sizes) if component_sizes else 0,
                "largest_component_volume_ml": (max(component_sizes) * voxel_volume_ml) if component_sizes else 0.0,
                "component_sizes_voxels": component_sizes,
                # "min_overlap_ratio": args.min_overlap,
            })

            print(f"Processed: {patient_id} {session}")

        except Exception as e:
            print(f"Error processing {mask_path}: {e}")

    df_cc = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df_cc.to_csv(args.out_csv, index=False)

    print(f"\nSaved to: {args.out_csv}")
    print(df_cc.head())


if __name__ == "__main__":
    main()
