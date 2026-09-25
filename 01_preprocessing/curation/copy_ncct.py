"""
copy_ncct.py
============
Copy skull-stripped NCCT files into the curated dataset under ses-01.

Source pattern::

    {src_dir}/sub-{pid}_ses-01_ncct_synthstrip.nii.gz

Destination pattern::

    {output_root}/sub-stroke_{pid}/ses-01/
        sub-stroke_{pid}_ses-01_ncct_skull_stripped.nii.gz

Standalone usage
----------------
    python copy_ncct.py --src-dir /path/to/ncct_skull_stripped \
                        --output-root /path/to/curated/raw_data_nifti
    python copy_ncct.py --dry-run

Pipeline usage
--------------
    from curation.copy_ncct import copy_ncct_files
"""

import re
import shutil
import argparse
from pathlib import Path

try:
    from config import NCCT_SKULL_STRIPPED_DIR, CURATED_ROOT
except ImportError:
    NCCT_SKULL_STRIPPED_DIR = None
    CURATED_ROOT = None

# sub-01_001_ses-01_ncct_synthstrip.nii.gz -> pid = 01_001
PATTERN = re.compile(r"^sub-(\d+_\d+)_ses-\d+_ncct_synthstrip\.nii\.gz$")


def copy_ncct_files(src_dir: Path, dest_root: Path, *, dry_run: bool = False):
    """Copy skull-stripped NCCT NIfTIs into the curated folder tree.

    Returns dict with counts: copied, skipped, exists.
    """
    if not src_dir.exists():
        raise FileNotFoundError(f"Source not found: {src_dir}")

    copied, skipped = 0, 0

    for src_file in sorted(src_dir.glob("sub-*_ncct_synthstrip.nii.gz")):
        m = PATTERN.match(src_file.name)
        if not m:
            print(f"[SKIP] name doesn't match pattern: {src_file.name}")
            skipped += 1
            continue

        pid = m.group(1)  # e.g. "01_001"

        dst_dir = dest_root / f"sub-stroke_{pid}" / "ses-01"
        dst_file = dst_dir / f"sub-stroke_{pid}_ses-01_ncct_skull_stripped.nii.gz"

        if dst_file.exists():
            print(f"[EXISTS] {dst_file}")
            skipped += 1
            continue

        if dry_run:
            print(f"[DRY] {src_file.name} -> {dst_file}")
            copied += 1
            continue

        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_file, dst_file)
        print(f"[COPIED] {src_file.name} -> {dst_file}")
        copied += 1

    print(f"\nDone. Copied={copied}  Skipped={skipped}")
    return {"copied": copied, "skipped": skipped}


def main():
    ap = argparse.ArgumentParser(
        description="Copy skull-stripped NCCT to curated dataset",
    )
    ap.add_argument(
        "--src-dir",
        type=Path,
        default=Path(NCCT_SKULL_STRIPPED_DIR) if NCCT_SKULL_STRIPPED_DIR else None,
        help="Directory with sub-*_ncct_synthstrip.nii.gz files",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=Path(CURATED_ROOT) / "raw_data_nifti" if CURATED_ROOT else None,
        help="Root of the curated raw_data_nifti tree",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print what would be copied without copying")
    args = ap.parse_args()

    if args.src_dir is None:
        ap.error("--src-dir is required (or set NCCT_SKULL_STRIPPED_DIR in config.py)")
    if args.output_root is None:
        ap.error("--output-root is required (or set CURATED_ROOT in config.py)")

    copy_ncct_files(args.src_dir, args.output_root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
