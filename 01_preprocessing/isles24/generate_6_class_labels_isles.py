"""generate_6_class_labels_isles.py

ISLES24 outcome-map generation (6 classes).

Same logic as SUS generate_6_class_labels.py. Numeric labels match the SUS
6-class map so downstream analyses written for SUS can be re-used on ISLES
without remapping:

    1  core_fi     : CTP core      & DWI infarct
    2  core_brain  : CTP core      & DWI healthy   (rescued core)
    3  pen_fi      : CTP penumbra  & DWI infarct
    4  pen_brain   : CTP penumbra  & DWI healthy   (rescued penumbra)
    5  CLB_brain   : CLB reference brain & DWI healthy, outside lesion safety
    6  nhb_fi      : DWI infarct & outside CTP lesion safety zone
"""

import os
import re
import glob
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import binary_dilation


# --------------------------------------------------
# Utility: 2D slice-wise dilation (AXIAL)
# --------------------------------------------------
def dilate_2d_per_slice(mask_3d, radius):
    """Apply 2D binary dilation independently per axial slice."""
    out = np.zeros_like(mask_3d, dtype=np.uint8)
    struct = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    for z in range(mask_3d.shape[0]):
        out[z] = binary_dilation(mask_3d[z], structure=struct)
    return out.astype(np.uint8)


# --------------------------------------------------
# Main processing function
# --------------------------------------------------
def generate_outcome_map_isles(ctp_path, clb_path, dwi_path, output_path, kernel_radius=1):

    # --- Load images ---
    ctp_img = sitk.ReadImage(ctp_path)
    clb_img = sitk.ReadImage(clb_path)
    dwi_img = sitk.ReadImage(dwi_path)

    # CLB is delivered on its own (DWI-native) grid even though the filename
    # carries `space-ncct`. Resample to the CTP/NCCT grid with NN so we can
    # index it element-wise alongside CTP and DWI.
    if (clb_img.GetSize() != ctp_img.GetSize()
            or clb_img.GetSpacing() != ctp_img.GetSpacing()
            or clb_img.GetDirection() != ctp_img.GetDirection()
            or clb_img.GetOrigin() != ctp_img.GetOrigin()):
        clb_img = sitk.Resample(
            clb_img,
            ctp_img,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            clb_img.GetPixelID(),
        )

    ctp_arr = sitk.GetArrayFromImage(ctp_img)
    clb_arr = sitk.GetArrayFromImage(clb_img)
    dwi_arr = sitk.GetArrayFromImage(dwi_img)

    if not (ctp_arr.shape == dwi_arr.shape == clb_arr.shape):
        raise RuntimeError(
            f"Shape mismatch after resample: CTP {ctp_arr.shape} "
            f"vs DWI {dwi_arr.shape} vs CLB {clb_arr.shape}"
        )

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

    # --- Combined exclusion zone (used for NHB) ---
    combined_safety = (core_safety == 1) | (pen_safety == 1)

    # --------------------------------------------------
    # 4. Visible buffer around CORE ONLY
    # --------------------------------------------------
    core_labels = (outcome_map == 1) | (outcome_map == 2)

    # Remove non-core tissue only inside CORE safety zone
    outcome_map[(core_safety == 1) & (~core_labels)] = 0

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
    # 7. Save output (preserve CTP geometry)
    # --------------------------------------------------
    out_img = sitk.GetImageFromArray(outcome_map)
    out_img.CopyInformation(ctp_img)
    sitk.WriteImage(out_img, output_path)


# --------------------------------------------------
# Path configuration
# --------------------------------------------------
ISLES_BASE = os.environ.get("ISLES24_WORK_ROOT", "/path/to/isles24-work")
ctp_dir = os.path.join(ISLES_BASE, "ISLES_filtered_ctp_labels_in_ncct")
clb_dir = os.path.join(ISLES_BASE, "CLB_filtered_segmentation")
dwi_dir = os.path.join(ISLES_BASE, "dwi_labels")
out_dir = os.path.join(ISLES_BASE, "6_Class_Labels")

os.makedirs(out_dir, exist_ok=True)

# Find CTP files
ctp_files = sorted(glob.glob(os.path.join(ctp_dir, "filtered_case_*_safe.nii.gz")))
print(f"Found {len(ctp_files)} ISLES CTP label files.")

written, skipped, failed = [], [], []

for ctp_file in ctp_files:
    fname = os.path.basename(ctp_file)

    # Parse 4-digit case ID: 'filtered_case_0001_safe.nii.gz' -> '0001'
    m = re.fullmatch(r"filtered_case_(\d+)_safe\.nii\.gz", fname)
    if not m:
        print(f"⚠️ Skip (cannot parse): {fname}")
        skipped.append(fname)
        continue
    case_num = m.group(1)
    pid = f"sub-stroke{case_num}"

    clb_path  = os.path.join(clb_dir,
                             f"filtered_{pid}_ses-02_space-ncct_dwi_synthseg.nii.gz")
    dwi_path  = os.path.join(dwi_dir,
                             f"{pid}_ses-02_space-ncct_lesion-msk.nii.gz")
    save_path = os.path.join(out_dir, f"{pid}_6class_map.nii.gz")

    missing = []
    if not os.path.exists(clb_path): missing.append(f"CLB -> {clb_path}")
    if not os.path.exists(dwi_path): missing.append(f"DWI -> {dwi_path}")
    if missing:
        print(f"⚠️ Skip {pid}: missing " + "; ".join(missing))
        skipped.append(pid)
        continue

    try:
        generate_outcome_map_isles(ctp_file, clb_path, dwi_path, save_path)
        print(f"✅ {pid}")
        written.append(pid)
    except Exception as e:
        print(f"❌ {pid}: {e}")
        failed.append((pid, str(e)))

print(f"\nDone. written={len(written)}  skipped={len(skipped)}  failed={len(failed)}")
if failed:
    print("\nFailed:")
    for pid, msg in failed:
        print(f"  {pid}: {msg}")
