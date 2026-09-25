"""
generate_6class_visualizations.py
==================================
Generate 6-class outcome map visualizations for all patients with complete data.

Creates multi-panel figures showing:
- NCCT background with 6-class overlay
- DWI background with 6-class overlay
- CTP background with 6-class overlay

Only processes patients with all required files: CTP, DWI, and 6-class labels.

Standalone usage
----------------
    python generate_6class_visualizations.py
    python generate_6class_visualizations.py --base-dir /data/reg --output-dir /out
    python generate_6class_visualizations.py --patients 01_001,01_002 --dry-run

Pipeline usage
--------------
    from analysis.generate_6class_visualizations import plot_6class_visualization
"""

import os
import glob
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
import matplotlib.pyplot as plt
import nibabel as nib
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from pathlib import Path
from tqdm import tqdm

try:
    from config import REG_ROOT, NCCT_SKULL_STRIPPED_DIR, SIX_CLASS_CLEANED_DIR
except ImportError:
    REG_ROOT = os.environ.get("SUS_WORK_ROOT", "/path/to/sus-work")
    NCCT_SKULL_STRIPPED_DIR = os.path.join(os.environ.get("SUS_WORK_ROOT", "/path/to/sus-work"), "ncct_skull_stripped")
    SIX_CLASS_CLEANED_DIR = os.path.join(REG_ROOT, "6_Class_Labels_Cleaned")

# =========================================================
# DEFAULT CONFIG
# =========================================================
_DEFAULT_BASE_DIR = REG_ROOT
_DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), os.pardir, "6class_visualizations"
)


def load_nifti(path):
    """Load NIfTI file using nibabel."""
    img = nib.load(str(path))
    data = img.get_fdata().astype(np.float32)
    # nibabel loads as (X, Y, Z) or (X, Y, Z, T)
    # Transpose to (Z, Y, X) or (T, Z, Y, X) to match expected format
    if data.ndim == 3:
        return data.transpose(2, 1, 0)  # (X,Y,Z) -> (Z,Y,X)
    elif data.ndim == 4:
        return data.transpose(3, 2, 1, 0)  # (X,Y,Z,T) -> (T,Z,Y,X)
    return data


def find_patients_with_complete_data(base_dir):
    """
    Find all patients who have CTP, DWI, and 6-class label files.

    Returns
    -------
    complete_patients : list of str
        Patient IDs with all required files
    """
    ctp_dir = os.path.join(base_dir, "ctp_in_ncct_windowed_and_skullstripped_native_mask")
    dwi_dir = os.path.join(base_dir, "DWI/preprocessed_skullstripped")
    label_dir = os.path.join(base_dir, "6_Class_Labels_Cleaned")

    # Find all 6-class label files
    label_files = glob.glob(os.path.join(label_dir, "sub-*_6class_map_cleaned.nii.gz"))

    complete_patients = []

    for label_file in label_files:
        # Extract patient ID (e.g., "01_001" from "sub-01_001_6class_map_cleaned.nii.gz")
        fname = os.path.basename(label_file)
        patient_id = fname.replace("sub-", "").replace("_6class_map_cleaned.nii.gz", "")

        # Check for CTP file
        ctp_file = os.path.join(ctp_dir, f"sub-{patient_id}_ses-01_space-ncct_ctp_mc_1fps_ss.nii.gz")
        if not os.path.exists(ctp_file):
            continue

        # Check for DWI file
        dwi_file = os.path.join(dwi_dir, f"sub-{patient_id}_dwi_in_ncct_ss.nii.gz")
        if not os.path.exists(dwi_file):
            continue

        complete_patients.append(patient_id)

    return sorted(complete_patients)


def plot_6class_visualization(patient_id, t_idx, base_dir, output_dir):
    """
    Create and save 6-class visualization figures for one patient.

    Parameters
    ----------
    patient_id : str
        Patient identifier (e.g., "01_001")
    t_idx : int
        Timepoint index for CTP visualization
    base_dir : str
        Base directory containing registration data
    output_dir : str
        Output directory for saved figures
    """
    # Path construction
    paths = {
        "ncct": os.path.join(NCCT_SKULL_STRIPPED_DIR, f"sub-{patient_id}_ses-01_ncct_synthstrip.nii.gz"),
        "dwi": f"{base_dir}/DWI/preprocessed_skullstripped/sub-{patient_id}_dwi_in_ncct_ss.nii.gz",
        "ctp": f"{base_dir}/ctp_in_ncct_windowed_and_skullstripped/sub-{patient_id}_ses-01_space-ncct_ctp_mc_1fps_ss.nii.gz",
        "outcome": os.path.join(SIX_CLASS_CLEANED_DIR, f"sub-{patient_id}_6class_map_cleaned.nii.gz"),
    }

    # Load data
    data = {}
    for key, path in paths.items():
        if os.path.exists(path):
            data[f"{key}_arr"] = load_nifti(path)
        else:
            print(f"  Warning: {key} file not found for {patient_id}")
            return False

    # Define colormap
    colors = ['none', 'red', 'blue', 'orange', 'lightgreen', 'green', 'yellow']
    outcome_cmap = ListedColormap(colors)

    legend_elements = [
        Line2D([0], [0], color='red', lw=4, label='Core → FI'),
        Line2D([0], [0], color='blue', lw=4, label='Core → Brain'),
        Line2D([0], [0], color='orange', lw=4, label='Penumbra → FI'),
        Line2D([0], [0], color='lightgreen', lw=4, label='Penumbra → Brain'),
        Line2D([0], [0], color='green', lw=4, label='CLB → Brain'),
        Line2D([0], [0], color='yellow', lw=4, label='NHB → FI')
    ]

    # Plot configurations: (array_key, title, vmin, vmax, is_4d)
    configs = [
        # ("ncct_arr", "NCCT", 0, 90, False),
        # ("dwi_arr", "DWI", 0, 1000, False),
        ("ctp_arr", f"CTP (t={t_idx})", 0, 255, True)
    ]

    # Create patient output directory
    patient_output_dir = os.path.join(output_dir, f"sub-{patient_id}")
    os.makedirs(patient_output_dir, exist_ok=True)

    for arr_key, title_suffix, vmin, vmax, is_4d in configs:
        bg_arr = data.get(arr_key)
        if bg_arr is None:
            continue

        num_slices = bg_arr.shape[0] if not is_4d else bg_arr.shape[1]
        cols = 10
        rows = int(np.ceil(num_slices / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(20, 2 * rows + 1), facecolor='black')
        fig.suptitle(f"Patient {patient_id} | {title_suffix}", color='white', fontsize=18, fontweight='bold')
        axes = axes.flatten()

        for i in range(num_slices):
            ax = axes[i]

            # Slice selection
            bg_slice = bg_arr[t_idx, i] if is_4d else bg_arr[i]

            # Base image
            ax.imshow(bg_slice, cmap="gray", vmin=vmin, vmax=vmax)

            # 6-class overlay
            if np.any(data["outcome_arr"][i] > 0):
                masked_map = np.ma.masked_where(data["outcome_arr"][i] == 0, data["outcome_arr"][i])
                ax.imshow(masked_map, cmap=outcome_cmap, alpha=0.6, vmin=0, vmax=6)

            ax.set_title(f"z={i}", fontsize=8, color='white')
            ax.axis('off')

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        # Add legend
        fig.legend(handles=legend_elements, loc='lower center', ncol=3,
                   bbox_to_anchor=(0.5, 0.01), frameon=False, labelcolor='white', fontsize=12)

        plt.tight_layout(rect=[0, 0.08, 1, 0.95])

        # Save figure
        modality = arr_key.replace("_arr", "")
        save_path = os.path.join(patient_output_dir, f"sub-{patient_id}_6class_{modality}.png")
        fig.savefig(save_path, dpi=150, facecolor='black', bbox_inches='tight')
        plt.close(fig)

    return True


def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(
        description="Generate 6-class outcome visualizations for patients with complete data."
    )

    parser.add_argument(
        "--base-dir",
        type=str,
        default=_DEFAULT_BASE_DIR,
        help="Base directory containing registration data"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=_DEFAULT_OUTPUT_DIR,
        help="Output directory for visualization figures"
    )

    parser.add_argument(
        "--timepoint",
        type=int,
        default=20,
        help="CTP timepoint index to visualize (default: 20)"
    )

    parser.add_argument(
        "--patients",
        type=str,
        default=None,
        help="Comma-separated list of patient IDs to process (e.g., '01_001,01_002'). Process all if not specified."
    )

    parser.add_argument(
        "--start-idx",
        type=int,
        default=None,
        help="Start index for batch processing (1-based)"
    )

    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="End index for batch processing (1-based)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List patients and exit without processing"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*70)
    print("6-Class Outcome Visualization Generator")
    print("="*70)

    # Find patients with complete data
    print("\nScanning for patients with complete data (CTP + DWI + 6-class labels)...")
    all_patients = find_patients_with_complete_data(args.base_dir)
    print(f"Found {len(all_patients)} patients with complete data")

    if len(all_patients) == 0:
        print("ERROR: No patients found with complete data!")
        return

    # Select patients to process
    if args.patients:
        # User-specified list
        selected_ids = [p.strip() for p in args.patients.split(',') if p.strip()]
        selected_ids = [p for p in selected_ids if p in all_patients]

        missing = set(args.patients.split(',')) - set(selected_ids)
        if missing:
            print(f"Warning: These patients don't have complete data: {missing}")
    else:
        # Range selection
        start = (args.start_idx - 1) if args.start_idx else 0
        end = args.end_idx if args.end_idx else len(all_patients)
        selected_ids = all_patients[start:end]

    print(f"\nSelected {len(selected_ids)} patients to process:")
    for i, pid in enumerate(selected_ids, 1):
        print(f"  {i:3d}. {pid}")

    if args.dry_run:
        print("\n(Dry run - exiting)")
        return

    # Process each patient
    print(f"\n{'='*70}")
    print("Starting visualization generation")
    print(f"{'='*70}\n")

    success_count = 0
    error_count = 0

    for i, patient_id in enumerate(tqdm(selected_ids, desc="Processing patients"), 1):
        try:
            success = plot_6class_visualization(
                patient_id=patient_id,
                t_idx=args.timepoint,
                base_dir=args.base_dir,
                output_dir=args.output_dir
            )

            if success:
                success_count += 1
            else:
                error_count += 1
                print(f"  Failed to process {patient_id}")

        except Exception as e:
            error_count += 1
            print(f"  Error processing {patient_id}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'='*70}")
    print("Processing completed")
    print(f"{'='*70}")
    print(f"Successful: {success_count}")
    print(f"Errors:     {error_count}")
    print(f"Total:      {len(selected_ids)}")
    print(f"\nFigures saved to: {args.output_dir}")
    print(f"\nEach patient folder contains 3 figures:")
    print(f"  - sub-{{patient_id}}_6class_ncct.png")
    print(f"  - sub-{{patient_id}}_6class_dwi.png")
    print(f"  - sub-{{patient_id}}_6class_ctp.png")


if __name__ == "__main__":
    main()
