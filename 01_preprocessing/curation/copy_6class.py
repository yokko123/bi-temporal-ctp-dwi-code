"""
copy_6class.py
==============
Copy cleaned 6-class bi-temporal lesion masks to the curated derivatives
folder.

Source pattern::

    {src_dir}/sub-{pid}_6class_map_cleaned.nii.gz

Destination pattern::

    {output_root}/sub-stroke_{pid}/ses-01/
        sub-stroke_{pid}_space-ncct_bi-temporal_lesion_msk.nii.gz

Standalone usage
----------------
    python copy_6class.py --src-dir /path/to/6_Class_Labels_Cleaned \
                          --output-root /path/to/curated/derivatives
    python copy_6class.py --dry-run

Pipeline usage
--------------
    from curation.copy_6class import copy_6class_files
"""

import re
import shutil
import argparse
from pathlib import Path

try:
    from config import SIX_CLASS_CLEANED_DIR, CURATED_ROOT
except ImportError:
    SIX_CLASS_CLEANED_DIR = None
    CURATED_ROOT = None

GLOB_PAT = "sub-*_6class_map_cleaned.nii.gz"
REGEX_PAT = r"sub-(\d+_\d+)_6class_map_cleaned\.nii\.gz"


def copy_6class_files(
    src_dir: Path,
    dest_root: Path,
    *,
    dry_run: bool = False,
    overwrite: bool = False,
):
    """Copy cleaned 6-class label NIfTIs into the curated folder tree.

    Returns dict with counts: copied, exists, skipped.
    """
    src_files = sorted(src_dir.glob(GLOB_PAT))
    print(f"Found {len(src_files)} 6-class label files\n")

    stats = {"copied": 0, "exists": 0, "skipped": 0}

    for src_file in src_files:
        m = re.match(REGEX_PAT, src_file.name)
        if not m:
            print(f"  [SKIP] No PID match: {src_file.name}")
            stats["skipped"] += 1
            continue

        pid = m.group(1)

        dst_dir = dest_root / f"sub-stroke_{pid}" / "ses-01"
        dst_file = dst_dir / f"sub-stroke_{pid}_space-ncct_bi-temporal_lesion_msk.nii.gz"

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
        description="Copy 6-class labels to curated derivatives",
    )
    ap.add_argument(
        "--src-dir",
        type=Path,
        default=Path(SIX_CLASS_CLEANED_DIR) if SIX_CLASS_CLEANED_DIR else None,
        help="Directory with sub-*_6class_map_cleaned.nii.gz files",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=Path(CURATED_ROOT) / "derivatives" if CURATED_ROOT else None,
        help="Root of the curated derivatives tree",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print actions without copying")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = ap.parse_args()

    if args.src_dir is None:
        ap.error("--src-dir is required (or set SIX_CLASS_CLEANED_DIR in config.py)")
    if args.output_root is None:
        ap.error("--output-root is required (or set CURATED_ROOT in config.py)")

    copy_6class_files(
        args.src_dir,
        args.output_root,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
