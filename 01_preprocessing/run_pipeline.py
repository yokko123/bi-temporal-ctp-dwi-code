#!/usr/bin/env python3
"""
Master CLI entry point for the SUS2025 preprocessing pipeline.

Run one or more steps for a given patient (or batch of patients).

Examples
--------
# Run motion correction + NCCT conversion for one patient:
    python run_pipeline.py --patients 01_001 --steps s01,s02

# Run full registration pipeline for multiple patients:
    python run_pipeline.py --patients 01_001,01_002 --steps s00,s01,s02,s03,s04,s05

# Run post-registration steps:
    python run_pipeline.py --patients 01_001 --steps p01,p02,p03,p04,p05,p06,p07

# Run skull stripping (NCCT-based):
    python run_pipeline.py --patients 01_001 --steps ss_ncct,ss_ctp_dwi

# Run curation:
    python run_pipeline.py --patients 01_001 --steps c_ncct,c_derivatives,c_6class

# List all available steps:
    python run_pipeline.py --list
"""

import argparse
import importlib
import sys

# ─── Registry of available steps ─────────────────────────────────────
STEPS = {
    # Registration
    "s00": ("registration.s00_extract_dicom_headers", "Extract DICOM headers to CSV"),
    "s01": ("registration.s01_ctp_motion_correction", "CTP motion correction + temporal resampling"),
    "s02": ("registration.s02_ncct_dicom_to_nifti", "Convert NCCT DICOM to NIfTI"),
    "s03": ("registration.s03_ctp_to_ncct", "Register CTP to NCCT space (+ labels, param maps)"),
    "s04": ("registration.s04_dwi_dicom_to_nifti", "Convert DWI DICOM to NIfTI"),
    "s05": ("registration.s05_dwi_adc_to_ncct", "Register DWI + ADC to NCCT space"),

    # Post-registration processing
    "p01": ("post_reg_processing.p01_filter_ctp_labels", "Filter CTP labels (remove brain class)"),
    "p02": ("post_reg_processing.p02_safety_margins", "Add safety margins to lesion labels"),
    "p03": ("post_reg_processing.p03_clb_filtering", "CLB filtering from SynthSeg"),
    "p04": ("post_reg_processing.p04_generate_6_class_labels", "Generate 6-class outcome labels"),
    "p05": ("post_reg_processing.p05_ctp_windowing", "CTP HU windowing"),
    "p06": ("post_reg_processing.p06_clean_ventricle_noise", "Clean ventricle noise from labels"),
    "p07": ("post_reg_processing.p07_remove_clb_ctp_overlap", "Remove CLB/CTP overlap"),

    # Skull stripping
    "ss_ncct": ("skull_stripping.skull_strip_ncct", "Skull-strip NCCT with SynthStrip"),
    "ss_ctp_dwi": ("skull_stripping.skull_strip_ctp_dwi", "Apply NCCT brain mask to CTP/DWI"),
    "ss_native": ("skull_stripping.skull_strip_ctp_native_masks", "Skull-strip CTP with native masks"),

    # Curation
    "c_ncct": ("curation.copy_ncct", "Copy NCCT to curated dataset"),
    "c_derivatives": ("curation.copy_derivatives", "Copy derivatives to curated dataset"),
    "c_6class": ("curation.copy_6class", "Copy 6-class labels to curated dataset"),
    "c_raw_ctp": ("curation.copy_raw_ctp", "Copy raw CTP to curated dataset"),

    # Analysis
    "a_voxel2": ("analysis.voxel_count_2_classes", "Voxel counts for 2-class labels"),
    "a_voxel6": ("analysis.voxel_count_6_classes", "Voxel counts for 6-class labels"),
    "a_voxel_dwi": ("analysis.voxel_count_dwi", "Voxel counts for DWI masks"),
    "a_cc_ctp": ("analysis.cc_ctp", "Connected-component analysis on CTP labels"),
    "a_cc_dwi": ("analysis.cc_dwi", "Connected-component analysis on DWI masks"),
    "a_metadata": ("analysis.collect_image_metadata", "Collect image metadata"),
    "a_vis": ("analysis.generate_6class_visualizations", "Generate 6-class visualizations"),
}


def list_steps():
    """Print all available pipeline steps."""
    print("\nAvailable pipeline steps:\n")
    current_group = None
    for key, (module, desc) in STEPS.items():
        group = module.split(".")[0]
        if group != current_group:
            current_group = group
            print(f"\n  [{group}]")
        print(f"    {key:16s} {desc}")
    print()


def run_step(step_key, extra_args=None):
    """Import and run a single pipeline step by its key."""
    if step_key not in STEPS:
        print(f"ERROR: Unknown step '{step_key}'")
        print(f"       Run with --list to see available steps.")
        return False

    module_path, desc = STEPS[step_key]
    print(f"\n{'='*60}")
    print(f"  Running: [{step_key}] {desc}")
    print(f"  Module:  {module_path}")
    print(f"{'='*60}\n")

    try:
        mod = importlib.import_module(module_path)
    except ImportError as e:
        print(f"ERROR: Could not import {module_path}: {e}")
        return False

    # Each module has a main() function that accepts sys.argv-style args
    if hasattr(mod, "main"):
        mod.main(extra_args or [])
    else:
        print(f"WARNING: {module_path} has no main() function, skipping.")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="SUS2025 Preprocessing Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--list", action="store_true", help="List all available steps")
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma-separated list of step keys to run (e.g. s01,s02,p01)",
    )
    parser.add_argument(
        "--patients",
        type=str,
        default=None,
        help="Comma-separated list of patient IDs (e.g. 01_001,01_002)",
    )
    parser.add_argument(
        "extra_args",
        nargs="*",
        help="Additional arguments passed to each step's main()",
    )

    args = parser.parse_args()

    if args.list:
        list_steps()
        return

    if not args.steps:
        parser.print_help()
        return

    step_keys = [s.strip() for s in args.steps.split(",")]
    patients = [p.strip() for p in args.patients.split(",")] if args.patients else [None]

    failed = []
    for patient in patients:
        for step_key in step_keys:
            extra = list(args.extra_args) if args.extra_args else []
            if patient:
                extra.extend(["--patient", patient])

            ok = run_step(step_key, extra)
            if not ok:
                failed.append((patient, step_key))

    print(f"\n{'='*60}")
    if failed:
        print(f"  Pipeline finished with {len(failed)} failure(s):")
        for pat, step in failed:
            print(f"    - patient={pat}, step={step}")
    else:
        total = len(step_keys) * len(patients)
        print(f"  Pipeline finished: {total} step(s) completed successfully.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
