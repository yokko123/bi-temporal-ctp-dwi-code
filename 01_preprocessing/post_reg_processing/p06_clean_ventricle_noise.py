"""
Post-registration step 6 – Remove ventricle noise from 6-class labels.

Any non-zero voxel in the 6-class outcome map that overlaps with the
lateral ventricles (SynthSeg labels 4 = Left-Lateral-Ventricle,
43 = Right-Lateral-Ventricle) is set to 0.  The result is saved as a
cleaned 6-class map.

Original script
---------------
    SUS2025-Preprocessing/scripts/clean_ventricle_noise.py
"""

import os
import glob
import argparse

import numpy as np
import SimpleITK as sitk

# ─── Config defaults (optional) ─────────────────────────────────────
try:
    from config import SYNTHSEG_DIR, SIX_CLASS_DIR, SIX_CLASS_CLEANED_DIR
except ImportError:
    SYNTHSEG_DIR = None
    SIX_CLASS_DIR = None
    SIX_CLASS_CLEANED_DIR = None

# SynthSeg ventricle label IDs
VENTRICLE_LABELS = [4, 43]


# =====================================================================
# Core processing function
# =====================================================================

def clean_ventricle_noise(sixclass_path, synthseg_path, output_path,
                          verbose=False):
    """Zero out 6-class labels inside the ventricle mask."""
    label_img = sitk.ReadImage(sixclass_path)
    seg_img = sitk.ReadImage(synthseg_path)

    label_arr = sitk.GetArrayFromImage(label_img)
    seg_arr = sitk.GetArrayFromImage(seg_img)

    # Build ventricle mask (left=4, right=43)
    ventricle_mask = np.isin(seg_arr, VENTRICLE_LABELS)

    # Count noisy voxels before cleaning
    noisy = (label_arr > 0) & ventricle_mask
    n_noisy = int(noisy.sum())

    if verbose:
        for cls in sorted(np.unique(label_arr[noisy])):
            cnt = int(((label_arr == cls) & ventricle_mask).sum())
            print(f"    class {cls}: {cnt} voxels in ventricle")

    # Zero out labels inside ventricles
    label_arr[ventricle_mask] = 0

    # Save
    out_img = sitk.GetImageFromArray(label_arr)
    out_img.CopyInformation(label_img)
    sitk.WriteImage(out_img, output_path, useCompression=True)

    return n_noisy


# =====================================================================
# CLI entry-point
# =====================================================================

def main(args=None):
    """Remove ventricle noise from all 6-class label files."""

    parser = argparse.ArgumentParser(
        description="Remove ventricle noise from 6-class labels"
    )
    parser.add_argument("--synthseg-dir", type=str, default=SYNTHSEG_DIR,
                        help="Directory with SynthSeg segmentation files")
    parser.add_argument("--input-dir", type=str, default=SIX_CLASS_DIR,
                        help="Directory with 6-class label files")
    parser.add_argument("--output-dir", type=str, default=SIX_CLASS_CLEANED_DIR,
                        help="Directory for cleaned output files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report counts without saving")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Per-class breakdown")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing outputs")

    args = parser.parse_args(args)

    if args.synthseg_dir is None or args.input_dir is None or args.output_dir is None:
        parser.error("--synthseg-dir, --input-dir, and --output-dir are required "
                     "(no config defaults available)")

    os.makedirs(args.output_dir, exist_ok=True)

    sixclass_files = sorted(
        glob.glob(os.path.join(args.input_dir, "sub-*_6class_map.nii.gz"))
    )
    print(f"Found {len(sixclass_files)} 6-class label files\n")

    total_cleaned = 0
    processed = 0
    skipped_missing = 0
    skipped_exists = 0

    for sc_path in sixclass_files:
        fname = os.path.basename(sc_path)
        # Extract PID: sub-01_001_6class_map.nii.gz -> 01_001
        pid = fname.replace("sub-", "").replace("_6class_map.nii.gz", "")

        # Match synthseg file
        seg_path = os.path.join(
            args.synthseg_dir, f"sub-{pid}_dwi_in_ncct_ss_synthseg.nii.gz"
        )
        out_path = os.path.join(
            args.output_dir, f"sub-{pid}_6class_map_cleaned.nii.gz"
        )

        if not os.path.exists(seg_path):
            print(f"[SKIP] sub-{pid}: SynthSeg not found")
            skipped_missing += 1
            continue

        if os.path.exists(out_path) and not args.overwrite:
            skipped_exists += 1
            continue

        if args.dry_run:
            # Still compute counts but don't save
            label_arr = sitk.GetArrayFromImage(sitk.ReadImage(sc_path))
            seg_arr = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
            ventricle_mask = np.isin(seg_arr, VENTRICLE_LABELS)
            n = int(((label_arr > 0) & ventricle_mask).sum())
            print(f"[DRY] sub-{pid}: {n} noisy voxels in ventricle")
            total_cleaned += n
            processed += 1
            continue

        n = clean_ventricle_noise(
            sc_path, seg_path, out_path, verbose=args.verbose
        )
        status = f"{n} voxels cleaned" if n > 0 else "clean"
        print(f"[OK]  sub-{pid}: {status}")
        total_cleaned += n
        processed += 1

    print(f"\nDone. Processed={processed}  Cleaned voxels={total_cleaned}  "
          f"Skipped(no synthseg)={skipped_missing}  "
          f"Skipped(exists)={skipped_exists}")


if __name__ == "__main__":
    main()
