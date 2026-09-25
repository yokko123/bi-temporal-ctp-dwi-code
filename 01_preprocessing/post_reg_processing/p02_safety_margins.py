"""
Post-registration step 2 – Add safety margins between lesion classes.

For each CTP label map the script:
    1. Extracts binary masks for brain (1), penumbra (2), core (3).
    2. Dilates penumbra and core with a configurable kernel.
    3. Subtracts the dilated region of the higher-priority class from
       the lower-priority one:
         • New brain  = old brain  − dilated penumbra
         • New penumbra = old penumbra − dilated core
    4. Reconstructs the label map and saves.

Original script
---------------
    SUS2025-Preprocessing/scripts/safety_margins_to_lesions/safety_margins.py
"""

import os
import glob
import argparse

import numpy as np
import SimpleITK as sitk

# ─── Config defaults (optional) ─────────────────────────────────────
try:
    from config import LABELS_IN_NCCT_DIR, SAFE_LABELS_DIR
except ImportError:
    LABELS_IN_NCCT_DIR = None
    SAFE_LABELS_DIR = None


# =====================================================================
# Core processing
# =====================================================================

def apply_safety_margins(input_path, output_path, kernel_radius=1):
    """Dilate penumbra/core and subtract overlaps to create safe margins."""
    if not os.path.exists(input_path):
        return

    # Load the label map
    label_img = sitk.ReadImage(input_path)
    labels = sitk.GetArrayFromImage(label_img)

    # 1. Extract individual binary masks
    brain = (labels == 1).astype(np.uint8)
    penumbra = (labels == 2).astype(np.uint8)
    core = (labels == 3).astype(np.uint8)

    # Convert to SITK images for dilation
    sitk_penumbra = sitk.GetImageFromArray(penumbra)
    sitk_core = sitk.GetImageFromArray(core)

    # 2. Dilation step
    # Dilate Penumbra to protect it from Brain
    dilated_penumbra = sitk.GetArrayFromImage(
        sitk.BinaryDilate(sitk_penumbra, [kernel_radius] * 3)
    )
    # Dilate Core to protect it from Penumbra
    dilated_core = sitk.GetArrayFromImage(
        sitk.BinaryDilate(sitk_core, [kernel_radius] * 3)
    )

    # 3. Subtraction (The "Safe Zone" Logic)
    # New Brain = Old Brain - Dilated Penumbra
    safe_brain = np.where((brain == 1) & (dilated_penumbra == 1), 0, brain)
    # New Penumbra = Old Penumbra - Dilated Core
    safe_penumbra = np.where((penumbra == 1) & (dilated_core == 1), 0, penumbra)

    # 4. Reconstruct Label Map (1=Brain, 2=Penumbra, 3=Core)
    final_labels = np.zeros_like(labels)
    final_labels[safe_brain == 1] = 1
    final_labels[safe_penumbra == 1] = 2
    final_labels[core == 1] = 3

    # Save
    out_img = sitk.GetImageFromArray(final_labels)
    out_img.CopyInformation(label_img)
    sitk.WriteImage(out_img, output_path)


def main(input_dir, output_dir, overwrite=False, kernel_radius=1):
    """Batch-process all .nii.gz files in *input_dir*."""
    os.makedirs(output_dir, exist_ok=True)

    # Find all .nii.gz files in the input folder
    file_pattern = os.path.join(input_dir, "*.nii.gz")
    files = glob.glob(file_pattern)

    print(f"Found {len(files)} files. Starting processing...")

    for in_file in sorted(files):
        filename = os.path.basename(in_file)
        # Create output name (e.g., sub-01_label_in_ncct_safe.nii.gz)
        out_filename = filename.replace(".nii.gz", "_safe.nii.gz")
        out_file = os.path.join(output_dir, out_filename)

        if not overwrite and os.path.exists(out_file):
            print(f"Skipping (exists): {filename}")
            continue

        try:
            apply_safety_margins(in_file, out_file, kernel_radius=kernel_radius)
            print(f"Processed: {filename}")
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

    print("Batch processing complete.")


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add safety margins between lesion classes by dilating "
                    "penumbra/core and subtracting overlaps.",
    )
    parser.add_argument(
        "--input-dir",
        default=LABELS_IN_NCCT_DIR,
        help="Directory with CTP label NIfTIs (default: config.LABELS_IN_NCCT_DIR)",
    )
    parser.add_argument(
        "--output-dir",
        default=SAFE_LABELS_DIR,
        help="Where to save safe-margin label NIfTIs (default: config.SAFE_LABELS_DIR)",
    )
    parser.add_argument(
        "--kernel-radius",
        type=int,
        default=1,
        help="Dilation kernel radius in voxels (default: 1)",
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

    main(args.input_dir, args.output_dir,
         overwrite=args.overwrite, kernel_radius=args.kernel_radius)
