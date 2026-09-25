"""
Post-registration step 1 – Remove "brain" label from CTP labels.

The CTP label maps contain three classes:
    1 = brain tissue
    2 = penumbra
    3 = ischemic core

This script filters them so only penumbra (2) and core (3) remain.
Everything else (background 0, brain 1) is set to 0.

Original script
---------------
    SUS2025-Preprocessing/scripts/filter_ctp_labels/ctp_labels_without_brain.py
"""

import os
import glob
import argparse

import numpy as np
import SimpleITK as sitk

# ─── Config defaults (optional) ─────────────────────────────────────
try:
    from config import LABELS_IN_NCCT_DIR, FILTERED_LABELS_DIR
except ImportError:
    LABELS_IN_NCCT_DIR = None
    FILTERED_LABELS_DIR = None

# Define labels to keep (ischemic core and penumbra)
KEEP_LABELS = [2, 3]


# =====================================================================
# Core processing
# =====================================================================

def process_ctp_labels(input_path, output_path):
    """Load a CTP label NIfTI, zero-out everything except KEEP_LABELS, save."""
    # Load the NIfTI label file
    img = sitk.ReadImage(input_path)
    arr = sitk.GetArrayFromImage(img)

    # Create a mask: Keep only pixels that are in our KEEP_LABELS list
    # Everything else (0, 1) becomes 0
    filtered_arr = np.where(np.isin(arr, KEEP_LABELS), arr, 0)

    # Convert back to SimpleITK image and preserve metadata (spacing, origin, direction)
    filtered_img = sitk.GetImageFromArray(filtered_arr.astype(np.uint8))
    filtered_img.CopyInformation(img)

    # Save the new file
    sitk.WriteImage(filtered_img, output_path)
    print(f"Processed: {os.path.basename(input_path)}")


def main(input_dir, output_dir, overwrite=False):
    """Batch-filter all CTP label files in *input_dir*."""
    os.makedirs(output_dir, exist_ok=True)

    # Find all label files in the directory
    label_files = glob.glob(os.path.join(input_dir, "*.nii.gz"))

    for file_path in label_files:
        filename = os.path.basename(file_path)
        # Output naming convention: filtered_sub-01_001_label_in_ncct.nii.gz
        out_file = os.path.join(output_dir, f"filtered_{filename}")

        if not overwrite and os.path.exists(out_file):
            print(f"Skipping (exists): {filename}")
            continue

        process_ctp_labels(file_path, out_file)

    print("\nFiltering complete. Only classes 2 and 3 remain.")


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove brain label (class 1) from CTP label maps, "
                    "keeping only penumbra (2) and core (3).",
    )
    parser.add_argument(
        "--input-dir",
        default=LABELS_IN_NCCT_DIR,
        help="Directory with raw CTP label NIfTIs (default: config.LABELS_IN_NCCT_DIR)",
    )
    parser.add_argument(
        "--output-dir",
        default=FILTERED_LABELS_DIR,
        help="Where to save filtered label NIfTIs (default: config.FILTERED_LABELS_DIR)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-process files even if output already exists",
    )
    args = parser.parse_args()

    if args.input_dir is None or args.output_dir is None:
        parser.error("--input-dir and --output-dir are required "
                      "(no config.py defaults found)")

    main(args.input_dir, args.output_dir, overwrite=args.overwrite)
