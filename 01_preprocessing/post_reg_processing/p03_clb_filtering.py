"""
Post-registration step 3 – Contralateral brain (CLB) extraction from SynthSeg.

Merges two original scripts:
  • clb_filter.py          – batch mode: iterate over a lesion-side JSON
  • clb_filter_per_patient.py – single-patient mode with argparse + custom labels

The segmentation keeps only the hemisphere **opposite** the lesion so that
downstream steps have a clean contralateral-brain mask.

Label sets
----------
    RIGHT_LABELS  – FreeSurfer/SynthSeg right-hemisphere structures
    LEFT_LABELS   – FreeSurfer/SynthSeg left-hemisphere structures

    If the lesion is on the *left*, we keep **right** labels (the CLB),
    and vice versa.

Original scripts
----------------
    SUS2025-Preprocessing/scripts/CLB_filtering/clb_filter.py
    SUS2025-Preprocessing/scripts/CLB_filtering/clb_filter_per_patient.py
"""

import os
import json
import argparse

import numpy as np
import SimpleITK as sitk

# ─── Config defaults (optional) ─────────────────────────────────────
try:
    from config import (
        SYNTHSEG_DIR,
        CLB_DIR,
        LESION_SIDE_JSON,
        LEFT_LABELS,
        RIGHT_LABELS,
    )
except ImportError:
    SYNTHSEG_DIR = None
    CLB_DIR = None
    LESION_SIDE_JSON = None
    # Default Label Sets (FreeSurfer/SynthSeg style)
    RIGHT_LABELS = [2, 3, 10, 11, 12, 13, 17, 18, 26, 28]
    LEFT_LABELS = [41, 42, 49, 50, 51, 52, 53, 54, 58, 60]


# =====================================================================
# Core processing – single file
# =====================================================================

def filter_segmentation(input_path, output_path, side, custom_labels=None):
    """Create a binary CLB mask from a SynthSeg segmentation.

    Parameters
    ----------
    input_path : str
        Path to input .nii.gz SynthSeg segmentation.
    output_path : str
        Where to save the filtered binary mask.
    side : str
        Lesion side ("left" or "right").  The *opposite* hemisphere
        labels are kept.
    custom_labels : list[int] or None
        If given, these labels override the side-based defaults.
    """
    if not os.path.exists(input_path):
        print(f"Error: File not found -> {input_path}")
        return

    # Load image
    img = sitk.ReadImage(input_path)
    arr = sitk.GetArrayFromImage(img)

    # Determine labels
    if custom_labels:
        keep_labels = custom_labels
        print(f"Using custom labels: {keep_labels}")
    elif side.lower() == "left":
        keep_labels = LEFT_LABELS
        print("Using default LEFT labels.")
    elif side.lower() == "right":
        keep_labels = RIGHT_LABELS
        print("Using default RIGHT labels.")
    else:
        print(f"Skipping {input_path}: Side is empty or unknown.")
        return

    # Apply binary mask
    mask = np.isin(arr, keep_labels)
    filtered_arr = np.where(mask, 1, 0).astype(np.uint8)

    # Save
    filtered_img = sitk.GetImageFromArray(filtered_arr)
    filtered_img.CopyInformation(img)
    sitk.WriteImage(filtered_img, output_path)
    print(f"Saved: {output_path}")


# =====================================================================
# Batch processing (from clb_filter.py)
# =====================================================================

def main(input_dir, output_dir, lesion_json, pid=None, side=None,
         overwrite=False, custom_labels=None):
    """Run CLB filtering in batch or single-patient mode.

    Parameters
    ----------
    input_dir : str
        Directory containing SynthSeg segmentation NIfTIs.
    output_dir : str
        Where to save filtered masks.
    lesion_json : str
        Path to JSON mapping patient_id → lesion side.
    pid : str or None
        If given, process only this patient (single-patient mode).
    side : str or None
        Lesion side override; required when *pid* is set and no JSON
        entry exists.
    overwrite : bool
        Re-process even if output already exists.
    custom_labels : list[int] or None
        Override default label lists.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Single-patient mode ──────────────────────────────────────────
    if pid is not None:
        filename = f"sub-{pid}_dwi_in_ncct_ss_synthseg.nii.gz"
        in_file = os.path.join(input_dir, filename)
        out_file = os.path.join(output_dir, f"filtered_{filename}")

        if not overwrite and os.path.exists(out_file):
            print(f"Skipping (exists): {filename}")
            return

        # Resolve side from JSON if not provided explicitly
        if side is None and lesion_json and os.path.exists(lesion_json):
            with open(lesion_json, "r") as f:
                lesion_mapping = json.load(f)
            side = lesion_mapping.get(pid, "")

        if not side:
            print(f"No side specified for patient {pid}. Skipping.")
            return

        filter_segmentation(in_file, out_file, side, custom_labels)
        return

    # ── Batch mode ───────────────────────────────────────────────────
    if not lesion_json or not os.path.exists(lesion_json):
        print("Error: --lesion-json is required for batch mode.")
        return

    with open(lesion_json, "r") as f:
        lesion_mapping = json.load(f)

    for patient_id, patient_side in lesion_mapping.items():
        filename = f"sub-{patient_id}_dwi_in_ncct_ss_synthseg.nii.gz"
        in_file = os.path.join(input_dir, filename)
        out_file = os.path.join(output_dir, f"filtered_{filename}")

        if not overwrite and os.path.exists(out_file):
            print(f"Skipping (exists): {filename}")
            continue

        filter_segmentation(in_file, out_file, patient_side, custom_labels)


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract contralateral-brain (CLB) mask from SynthSeg "
                    "segmentation.  Runs in batch mode by default; use --pid "
                    "for a single patient.",
    )
    parser.add_argument(
        "--input-dir",
        default=SYNTHSEG_DIR,
        help="Directory with SynthSeg segmentation NIfTIs "
             "(default: config.SYNTHSEG_DIR)",
    )
    parser.add_argument(
        "--output-dir",
        default=CLB_DIR,
        help="Where to save filtered CLB masks (default: config.CLB_DIR)",
    )
    parser.add_argument(
        "--lesion-json",
        default=LESION_SIDE_JSON,
        help="JSON file mapping patient ID → lesion side "
             "(default: config.LESION_SIDE_JSON)",
    )
    parser.add_argument(
        "--pid",
        default=None,
        help="Process only this patient ID (e.g. 01_001).  "
             "Requires --side or a matching JSON entry.",
    )
    parser.add_argument(
        "--side",
        choices=["left", "right"],
        default=None,
        help="Lesion side (overrides JSON lookup when using --pid)",
    )
    parser.add_argument(
        "--labels",
        type=int,
        nargs="+",
        default=None,
        help="Custom label IDs to keep (overrides side-based defaults)",
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

    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        lesion_json=args.lesion_json,
        pid=args.pid,
        side=args.side,
        overwrite=args.overwrite,
        custom_labels=args.labels,
    )
