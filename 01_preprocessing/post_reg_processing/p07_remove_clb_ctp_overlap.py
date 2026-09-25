"""
Post-registration step 7 – Remove CLB / CTP penumbra-core overlap.

CLB voxels that overlap with CTP penumbra (2) or core (3) are removed
so that CLB contains only healthy contralateral tissue.  A configurable
dilation margin creates a safety gap around lesion boundaries.

The CLB file is overwritten in-place (backed up with .bak unless
``--no-backup`` is given).

Original script
---------------
    SUS2025-Preprocessing/scripts/remove_clb_ctp_overlap.py
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_dilation

# ─── Config defaults (optional) ─────────────────────────────────────
try:
    from config import CLB_DIR, SAFE_LABELS_DIR
except ImportError:
    CLB_DIR = None
    SAFE_LABELS_DIR = None

DEFAULT_PATIENTS = ["01_046", "01_047", "01_076"]


# =====================================================================
# Utility: 2D slice-wise dilation
# =====================================================================

def dilate_2d_per_slice(mask_3d, radius):
    """
    Apply 2D binary dilation independently on each axial slice.
    Avoids z-leakage in anisotropic volumes.
    """
    out = np.zeros_like(mask_3d, dtype=bool)
    struct = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    for z in range(mask_3d.shape[0]):
        if mask_3d[z].any():
            out[z] = binary_dilation(mask_3d[z], structure=struct)
    return out


# =====================================================================
# Core processing function
# =====================================================================

def remove_overlap(pid, clb_dir, ctp_labels_dir, margin=3,
                   dry_run=False, no_backup=False):
    """Remove CLB voxels overlapping with dilated CTP lesion mask."""

    clb_dir = Path(clb_dir)
    ctp_labels_dir = Path(ctp_labels_dir)

    clb_path = clb_dir / f"filtered_sub-{pid}_dwi_in_ncct_ss_synthseg.nii.gz"
    ctp_path = ctp_labels_dir / f"filtered_sub-{pid}_label_in_ncct_safe.nii.gz"

    if not clb_path.exists():
        print(f"  [SKIP] CLB not found: {clb_path.name}")
        return False
    if not ctp_path.exists():
        print(f"  [SKIP] CTP not found: {ctp_path.name}")
        return False

    # Load
    clb_img = sitk.ReadImage(str(clb_path))
    ctp_img = sitk.ReadImage(str(ctp_path))

    clb_arr = sitk.GetArrayFromImage(clb_img)
    ctp_arr = sitk.GetArrayFromImage(ctp_img)

    # Lesion mask: penumbra (2) or core (3)
    lesion_mask = (ctp_arr >= 2)

    # Dilate lesion mask to create safety gap
    if margin > 0:
        exclusion_zone = dilate_2d_per_slice(lesion_mask, margin)
        print(f"  Margin: {margin} voxels (2D slice-wise dilation)")
    else:
        exclusion_zone = lesion_mask

    # Find CLB voxels inside the exclusion zone
    overlap = (clb_arr > 0) & exclusion_zone
    n_overlap = int(overlap.sum())
    n_clb_before = int((clb_arr > 0).sum())

    print(f"  CLB voxels before: {n_clb_before}")
    print(f"  Voxels to remove (overlap + margin): {n_overlap}")
    print(f"  CLB voxels after:  {n_clb_before - n_overlap}")

    if n_overlap == 0:
        print("  No overlap — skipping.")
        return True

    if dry_run:
        print(f"  [DRY-RUN] Would remove {n_overlap} voxels")
        return True

    # Backup original
    if not no_backup:
        bak_path = clb_path.with_suffix(".nii.gz.bak")
        shutil.copyfile(clb_path, bak_path)
        print(f"  Backup: {bak_path.name}")

    # Remove overlap
    clb_arr[overlap] = 0

    # Save
    out_img = sitk.GetImageFromArray(clb_arr)
    out_img.CopyInformation(clb_img)
    sitk.WriteImage(out_img, str(clb_path), useCompression=True)
    print(f"  Saved cleaned CLB: {clb_path.name}")
    return True


# =====================================================================
# CLI entry-point
# =====================================================================

def main(args=None):
    """Remove CLB/CTP penumbra-core overlap for selected patients."""

    ap = argparse.ArgumentParser(
        description="Remove CLB/CTP penumbra-core overlap"
    )
    ap.add_argument("--clb-dir", type=str, default=CLB_DIR,
                    help="Directory with CLB segmentation files")
    ap.add_argument("--ctp-labels-dir", type=str, default=SAFE_LABELS_DIR,
                    help="Directory with CTP safety-margin labels")
    ap.add_argument("--patients", type=str, default=None,
                    help="Comma-separated patient IDs "
                         "(default: 01_046,01_047,01_076)")
    ap.add_argument("--margin", type=int, default=3,
                    help="Safety gap in voxels around penumbra/core "
                         "(2D dilation radius, default: 3)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report overlap without modifying files")
    ap.add_argument("--no-backup", action="store_true",
                    help="Skip creating .bak backup files")

    args = ap.parse_args(args)

    if args.clb_dir is None or args.ctp_labels_dir is None:
        ap.error("--clb-dir and --ctp-labels-dir are required "
                 "(no config defaults available)")

    patients = (
        [p.strip() for p in args.patients.split(",")]
        if args.patients
        else DEFAULT_PATIENTS
    )

    print(f"Processing {len(patients)} patients (margin={args.margin} voxels)\n")

    for pid in patients:
        print(f"[{pid}]")
        remove_overlap(
            pid,
            clb_dir=args.clb_dir,
            ctp_labels_dir=args.ctp_labels_dir,
            margin=args.margin,
            dry_run=args.dry_run,
            no_backup=args.no_backup,
        )
        print()

    print("Done.")


if __name__ == "__main__":
    main()
