"""
copy_derivatives.py
===================
Copy registered CTP, DWI, ADC, and their labels into the curated
derivatives folder.

Destination structure::

    derivatives/
      sub-stroke_{pid}/
        ses-01/
          sub-stroke_{pid}_ses-01_space-ncct_ctp.nii.gz
          sub-stroke_{pid}_ses-01_space-ncct_ctp_lesion_msk_3class.nii.gz
        ses-02/
          sub-stroke_{pid}_ses-02_space-ncct_dwi.nii.gz
          sub-stroke_{pid}_ses-02_space-ncct_adc.nii.gz
          sub-stroke_{pid}_ses-02_space-ncct_dwi_lesion_msk.nii.gz

Standalone usage
----------------
    python copy_derivatives.py --output-root /path/to/curated/derivatives
    python copy_derivatives.py --dry-run --modality ctp,dwi

Pipeline usage
--------------
    from curation.copy_derivatives import copy_derivative_files
"""

import re
import shutil
import argparse
from pathlib import Path

try:
    from config import (
        CTP_WINDOWED_DIR,
        LABELS_IN_NCCT_DIR,
        DWI_SS_DIR,
        ADC_REG_DIR,
        DWI_MASKS_DIR,
        CURATED_ROOT,
    )
except ImportError:
    CTP_WINDOWED_DIR = None
    LABELS_IN_NCCT_DIR = None
    DWI_SS_DIR = None
    ADC_REG_DIR = None
    DWI_MASKS_DIR = None
    CURATED_ROOT = None

# ==========================================================
# Source filename patterns → PID extraction
# ==========================================================
# Each entry: (source_dir_key, glob_pattern, regex, session, dest_filename_template)
_DEFAULT_SRC = {
    "ctp": CTP_WINDOWED_DIR,
    "ctp_label": LABELS_IN_NCCT_DIR,
    "dwi": DWI_SS_DIR,
    "adc": ADC_REG_DIR,
    "dwi_mask": DWI_MASKS_DIR,
}

FILE_MAP = [
    # --- ses-01 ---
    (
        "ctp",
        "sub-*_ses-01_space-ncct_ctp_mc_1fps_ss.nii.gz",
        r"sub-(\d+_\d+)_ses-01_space-ncct_ctp_mc_1fps_ss\.nii\.gz",
        "ses-01",
        "sub-stroke_{pid}_ses-01_space-ncct_ctp.nii.gz",
    ),
    (
        "ctp_label",
        "sub-*_label_in_ncct.nii.gz",
        r"sub-(\d+_\d+)_label_in_ncct\.nii\.gz",
        "ses-01",
        "sub-stroke_{pid}_ses-01_space-ncct_ctp_lesion_msk_3class.nii.gz",
    ),
    # --- ses-02 ---
    (
        "dwi",
        "sub-*_dwi_in_ncct_ss.nii.gz",
        r"sub-(\d+_\d+)_dwi_in_ncct_ss\.nii\.gz",
        "ses-02",
        "sub-stroke_{pid}_ses-02_space-ncct_dwi.nii.gz",
    ),
    (
        "adc",
        "sub-*_adc_in_ncct_ss.nii.gz",
        r"sub-(\d+_\d+)_adc_in_ncct_ss\.nii\.gz",
        "ses-02",
        "sub-stroke_{pid}_ses-02_space-ncct_adc.nii.gz",
    ),
    (
        "dwi_mask",
        "sub-*_dwi_mask_in_ncct.nii.gz",
        r"sub-(\d+_\d+)_dwi_mask_in_ncct\.nii\.gz",
        "ses-02",
        "sub-stroke_{pid}_ses-02_space-ncct_dwi_lesion_msk.nii.gz",
    ),
]


def copy_derivative_files(
    src_dirs: dict,
    dest_root: Path,
    *,
    dry_run: bool = False,
    overwrite: bool = False,
    selected: set | None = None,
):
    """Copy registered derivative NIfTIs into the curated folder tree.

    Parameters
    ----------
    src_dirs : dict
        Mapping modality key -> Path for source directories.
    dest_root : Path
        Root of the curated derivatives tree.
    dry_run : bool
        Print actions without copying.
    overwrite : bool
        Overwrite existing files.
    selected : set or None
        If given, only process these modality keys.

    Returns dict with counts.
    """
    stats = {"copied": 0, "skipped": 0, "missing": 0, "exists": 0}

    for src_key, glob_pat, regex_pat, session, dest_template in FILE_MAP:
        if selected and src_key not in selected:
            continue

        src_dir = src_dirs.get(src_key)
        if src_dir is None or not Path(src_dir).exists():
            print(f"[WARN] Source dir not found for '{src_key}': {src_dir}")
            continue

        src_dir = Path(src_dir)
        src_files = sorted(src_dir.glob(glob_pat))
        print(f"\n--- {src_key} ({session}) : {len(src_files)} files ---")

        for src_file in src_files:
            m = re.match(regex_pat, src_file.name)
            if not m:
                print(f"  [SKIP] No PID match: {src_file.name}")
                stats["skipped"] += 1
                continue

            pid = m.group(1)  # e.g. "01_001"

            dst_dir = dest_root / f"sub-stroke_{pid}" / session
            dst_file = dst_dir / dest_template.format(pid=pid)

            if dst_file.exists() and not overwrite:
                stats["exists"] += 1
                continue

            if dry_run:
                print(f"  [DRY] {src_file.name} -> {dst_file}")
                stats["copied"] += 1
                continue

            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_file, dst_file)
            print(f"  [OK] {pid} -> {dst_file.name}")
            stats["copied"] += 1

    print(
        f"\nDone. Copied={stats['copied']}  Already exists={stats['exists']}  "
        f"Skipped={stats['skipped']}  Missing={stats['missing']}"
    )
    return stats


def main():
    ap = argparse.ArgumentParser(
        description="Copy registered derivatives to curated folder",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=Path(CURATED_ROOT) / "derivatives" if CURATED_ROOT else None,
        help="Root of the curated derivatives tree",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print actions without copying")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    ap.add_argument(
        "--modality",
        type=str,
        default=None,
        help="Comma-separated modalities to copy (e.g. 'ctp,dwi'). "
        "Options: ctp, ctp_label, dwi, adc, dwi_mask. Default: all",
    )
    args = ap.parse_args()

    if args.output_root is None:
        ap.error("--output-root is required (or set CURATED_ROOT in config.py)")

    # Parse selected modalities
    selected = None
    if args.modality:
        selected = set(m.strip() for m in args.modality.split(","))
        valid = {entry[0] for entry in FILE_MAP}
        invalid = selected - valid
        if invalid:
            ap.error(f"Unknown modalities: {invalid}. Valid: {valid}")
        print(f"Selected modalities: {selected}")

    # Build source dirs from config defaults
    src_dirs = {k: v for k, v in _DEFAULT_SRC.items() if v is not None}

    copy_derivative_files(
        src_dirs,
        args.output_root,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        selected=selected,
    )


if __name__ == "__main__":
    main()
