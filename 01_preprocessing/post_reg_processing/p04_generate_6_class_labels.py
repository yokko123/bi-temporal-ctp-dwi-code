"""
Post-registration step 4 – Generate 6-class bi-temporal outcome maps.

The outcome map combines three inputs:
    • CTP labels (penumbra / core)
    • CLB binary mask (contralateral brain)
    • DWI binary mask (final infarct)

Output classes
--------------
    1 = Core_fi     – core region that IS in the final infarct
    2 = Core_brain  – core region that is NOT in the final infarct
    3 = Pen_fi      – penumbra region that IS in the final infarct
    4 = Pen_brain   – penumbra region that is NOT in the final infarct
    5 = CLB_brain   – contralateral brain, no infarct, outside safety zone
    6 = NHB_fi      – non-hypo-perfused brain tissue IN the final infarct

Safety zones (2D slice-wise dilation) prevent class leakage at
tissue boundaries.

Original script
---------------
    SUS2025-Preprocessing/scripts/generate_6_class_labels/generate_6_class_labels.py
"""

import os
import glob
import argparse

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_dilation

# ─── Config defaults (optional) ─────────────────────────────────────
try:
    from config import (
        FILTERED_LABELS_DIR,
        CLB_DIR,
        DWI_MASKS_DIR,
        SIX_CLASS_DIR,
    )
except ImportError:
    FILTERED_LABELS_DIR = None
    CLB_DIR = None
    DWI_MASKS_DIR = None
    SIX_CLASS_DIR = None


# =====================================================================
# Utility: 2D slice-wise dilation (AXIAL)
# =====================================================================

def dilate_2d_per_slice(mask_3d, radius):
    """
    Apply 2D binary dilation independently on each axial slice.
    This avoids z-leakage in anisotropic medical volumes.
    """
    out = np.zeros_like(mask_3d, dtype=np.uint8)
    struct = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)

    for z in range(mask_3d.shape[0]):
        out[z] = binary_dilation(mask_3d[z], structure=struct)

    return out.astype(np.uint8)


# =====================================================================
# Main processing function
# =====================================================================

def generate_outcome_map(ctp_path, clb_path, dwi_path, output_path,
                         kernel_radius=1):
    """Build a 6-class outcome map for one patient."""

    # --- Load images ---
    ctp_img = sitk.ReadImage(ctp_path)
    clb_img = sitk.ReadImage(clb_path)
    dwi_img = sitk.ReadImage(dwi_path)

    ctp_arr = sitk.GetArrayFromImage(ctp_img)
    clb_arr = sitk.GetArrayFromImage(clb_img)
    dwi_arr = sitk.GetArrayFromImage(dwi_img)

    # --------------------------------------------------
    # 1. Initialize outcome map
    # --------------------------------------------------
    outcome_map = np.zeros_like(ctp_arr, dtype=np.uint8)

    # --------------------------------------------------
    # 2. Core / Penumbra outcomes (labels 1–4)
    # --------------------------------------------------
    outcome_map[(ctp_arr == 3) & (dwi_arr == 1)] = 1  # Core_fi
    outcome_map[(ctp_arr == 3) & (dwi_arr == 0)] = 2  # Core_brain
    outcome_map[(ctp_arr == 2) & (dwi_arr == 1)] = 3  # Pen_fi
    outcome_map[(ctp_arr == 2) & (dwi_arr == 0)] = 4  # Pen_brain

    # --------------------------------------------------
    # 3. Safety zones (2D slice-wise)
    # --------------------------------------------------

    # --- Core outcome safety zone (final core) ---
    core_outcome_mask = ((outcome_map == 1) | (outcome_map == 2)).astype(np.uint8)
    core_safety = dilate_2d_per_slice(core_outcome_mask, kernel_radius)

    # --- Penumbra safety zone (acute penumbra) ---
    pen_mask = (ctp_arr == 2).astype(np.uint8)
    pen_safety = dilate_2d_per_slice(pen_mask, kernel_radius)

    # --- Combined exclusion zone (for NHB / CLB only) ---
    combined_safety = (core_safety == 1) | (pen_safety == 1)

    # --------------------------------------------------
    # 4. Visible buffer around CORE ONLY
    #    (do NOT erase penumbra globally)
    # --------------------------------------------------
    core_labels = (outcome_map == 1) | (outcome_map == 2)

    # Remove non-core tissue only inside CORE safety zone
    outcome_map[
        (core_safety == 1) & (~core_labels)
    ] = 0

    # --------------------------------------------------
    # 5. CLB_brain (label 5)
    # --------------------------------------------------
    outcome_map[
        (clb_arr == 1) &
        (dwi_arr == 0) &
        (combined_safety == 0)
    ] = 5

    # --------------------------------------------------
    # 6. NHB_fi (label 6)
    # --------------------------------------------------
    outcome_map[
        (dwi_arr == 1) &
        (ctp_arr != 2) &
        (ctp_arr != 3) &
        (combined_safety == 0)
    ] = 6

    # --------------------------------------------------
    # 7. Save output
    # --------------------------------------------------
    out_img = sitk.GetImageFromArray(outcome_map)
    out_img.CopyInformation(ctp_img)
    sitk.WriteImage(out_img, output_path)


def main(ctp_labels_dir, clb_dir, dwi_masks_dir, output_dir,
         overwrite=False, kernel_radius=1):
    """Batch-generate 6-class outcome maps for all patients."""
    os.makedirs(output_dir, exist_ok=True)

    # Find files
    ctp_files = sorted(
        glob.glob(os.path.join(ctp_labels_dir, "filtered_sub-*_label_in_ncct.nii.gz"))
    )

    print(f"Processing {len(ctp_files)} patients...")

    for ctp_file in ctp_files:
        fname = os.path.basename(ctp_file)

        # Parse patient ID: everything between 'sub-' and '_label'
        # Example: 'filtered_sub-01_001_label_in_ncct_safe.nii.gz' -> '01_001'
        try:
            p_id = fname.split("sub-")[1].split("_label")[0]
        except IndexError:
            print(f"Could not parse ID from filename: {fname}")
            continue

        save_path = os.path.join(output_dir, f"sub-{p_id}_6class_map.nii.gz")

        if not overwrite and os.path.exists(save_path):
            print(f"Skipping (exists): sub-{p_id}")
            continue

        # Construct matching paths using the specific ID format
        clb_path = os.path.join(
            clb_dir, f"filtered_sub-{p_id}_dwi_in_ncct_ss_synthseg.nii.gz"
        )
        dwi_path = os.path.join(
            dwi_masks_dir, f"sub-{p_id}_dwi_mask_in_ncct.nii.gz"
        )

        if os.path.exists(clb_path) and os.path.exists(dwi_path):
            try:
                generate_outcome_map(
                    ctp_file, clb_path, dwi_path, save_path,
                    kernel_radius=kernel_radius,
                )
                print(f"Success: sub-{p_id}")
            except Exception as e:
                print(f"Error on sub-{p_id}: {e}")
        else:
            print(f"Skip sub-{p_id}: Missing files")
            if not os.path.exists(clb_path):
                print(f"   - Missing CLB: {clb_path}")
            if not os.path.exists(dwi_path):
                print(f"   - Missing DWI: {dwi_path}")

    print("\nDone.")


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate 6-class bi-temporal outcome maps combining "
                    "CTP labels, CLB mask, and DWI mask.",
    )
    parser.add_argument(
        "--ctp-labels-dir",
        default=FILTERED_LABELS_DIR,
        help="Directory with filtered CTP label NIfTIs "
             "(default: config.FILTERED_LABELS_DIR)",
    )
    parser.add_argument(
        "--clb-dir",
        default=CLB_DIR,
        help="Directory with CLB binary mask NIfTIs "
             "(default: config.CLB_DIR)",
    )
    parser.add_argument(
        "--dwi-masks-dir",
        default=DWI_MASKS_DIR,
        help="Directory with DWI binary mask NIfTIs "
             "(default: config.DWI_MASKS_DIR)",
    )
    parser.add_argument(
        "--output-dir",
        default=SIX_CLASS_DIR,
        help="Where to save 6-class outcome maps "
             "(default: config.SIX_CLASS_DIR)",
    )
    parser.add_argument(
        "--kernel-radius",
        type=int,
        default=1,
        help="Safety-zone dilation radius in voxels (default: 1)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-process files even if output already exists",
    )
    args = parser.parse_args()

    if (args.ctp_labels_dir is None or args.clb_dir is None
            or args.dwi_masks_dir is None or args.output_dir is None):
        parser.error("All directory arguments are required "
                      "(no config.py defaults found)")

    main(
        ctp_labels_dir=args.ctp_labels_dir,
        clb_dir=args.clb_dir,
        dwi_masks_dir=args.dwi_masks_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        kernel_radius=args.kernel_radius,
    )
