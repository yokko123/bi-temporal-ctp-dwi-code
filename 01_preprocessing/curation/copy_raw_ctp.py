"""
copy_raw_ctp.py
===============
Copy raw motion-corrected CTP and raw CTP labels to the curated
raw_data_nifti folder.

Source patterns::

    {ctp_mc_dir}/sub-{pid}_ses-01_ctp_mc_1fps.nii.gz
    {labels_dir}/sub-{pid}_label_ctp.nii.gz

Destination (raw_data_nifti/sub-stroke_{pid}/ses-01/)::

    sub-stroke_{pid}_ctp_motion_corrected_1fps.nii.gz
    sub-stroke_{pid}_ctp_lesion_msk.nii.gz

Standalone usage
----------------
    python copy_raw_ctp.py --output-root /path/to/curated/raw_data_nifti
    python copy_raw_ctp.py --dry-run

Pipeline usage
--------------
    from curation.copy_raw_ctp import copy_raw_ctp_files
"""

import re
import shutil
import argparse
from pathlib import Path

try:
    from config import CTP_MC_DIR, RAW_LABELS_CTP_DIR, CURATED_ROOT
except ImportError:
    CTP_MC_DIR = None
    RAW_LABELS_CTP_DIR = None
    CURATED_ROOT = None

# Each entry: (source_dir, glob, regex, dest_template)
_DEFAULT_FILE_MAP = [
    (
        CTP_MC_DIR,
        "sub-*_ses-01_ctp_mc_1fps.nii.gz",
        r"sub-(\d+_\d+)_ses-01_ctp_mc_1fps\.nii\.gz",
        "sub-stroke_{pid}_ctp_motion_corrected_1fps.nii.gz",
    ),
    (
        RAW_LABELS_CTP_DIR,
        "sub-*_label_ctp.nii.gz",
        r"sub-(\d+_\d+)_label_ctp\.nii\.gz",
        "sub-stroke_{pid}_ctp_lesion_msk.nii.gz",
    ),
]


def copy_raw_ctp_files(
    file_map: list,
    dest_root: Path,
    *,
    dry_run: bool = False,
    overwrite: bool = False,
):
    """Copy raw CTP NIfTIs and labels into the curated folder tree.

    Parameters
    ----------
    file_map : list of tuples
        Each tuple: (source_dir, glob_pattern, regex_pattern, dest_template).
    dest_root : Path
        Root of the curated raw_data_nifti tree.
    dry_run : bool
        Print actions without copying.
    overwrite : bool
        Overwrite existing files.

    Returns dict with counts.
    """
    stats = {"copied": 0, "exists": 0, "skipped": 0}

    for src_dir, glob_pat, regex_pat, dest_tmpl in file_map:
        if src_dir is None:
            print(f"[WARN] Source dir is None – skipping pattern {glob_pat}")
            continue

        src_dir = Path(src_dir)
        if not src_dir.exists():
            print(f"[WARN] Source dir not found: {src_dir}")
            continue

        src_files = sorted(src_dir.glob(glob_pat))
        print(f"\n--- {src_dir.name}: {len(src_files)} files ---")

        for src_file in src_files:
            m = re.match(regex_pat, src_file.name)
            if not m:
                print(f"  [SKIP] No PID match: {src_file.name}")
                stats["skipped"] += 1
                continue

            pid = m.group(1)
            dst_dir = dest_root / f"sub-stroke_{pid}" / "ses-01"
            dst_file = dst_dir / dest_tmpl.format(pid=pid)

            if dst_file.exists() and not overwrite:
                stats["exists"] += 1
                continue

            if dry_run:
                print(f"  [DRY] {src_file.name} -> {dst_file.name}")
                stats["copied"] += 1
                continue

            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_file, dst_file)
            print(f"  [OK] {pid} -> {dst_file.name}")
            stats["copied"] += 1

    print(
        f"\nDone. Copied={stats['copied']}  Already exists={stats['exists']}  "
        f"Skipped={stats['skipped']}"
    )
    return stats


def main():
    ap = argparse.ArgumentParser(
        description="Copy raw CTP and labels to curated raw_data_nifti",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=Path(CURATED_ROOT) / "raw_data_nifti" if CURATED_ROOT else None,
        help="Root of the curated raw_data_nifti tree",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print actions without copying")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = ap.parse_args()

    if args.output_root is None:
        ap.error("--output-root is required (or set CURATED_ROOT in config.py)")

    copy_raw_ctp_files(
        _DEFAULT_FILE_MAP,
        args.output_root,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
