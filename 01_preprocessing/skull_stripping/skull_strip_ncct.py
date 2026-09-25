"""
Batch skull-strip NCCT volumes using synthstrip-singularity.

Refactored from:
    SUS2025-Preprocessing/scripts/skull_stripping/skull_stripping_ncct.py

For each NIfTI in the input directory the script calls
``synthstrip-singularity`` to produce a skull-stripped image and a binary
brain mask.  If the ``-m`` flag is not supported by the local wrapper it
retries without the mask output flag.

Entry point
-----------
    synthstrip_batch(ncct_dir, ...)  – process every file in a directory
    __main__ argparse block          – CLI wrapper
"""

import os
import subprocess
import argparse
from pathlib import Path

# ─── Config defaults (optional) ─────────────────────────────────────
try:
    from config import (
        NCCT_NIFTI_DIR,
        NCCT_SKULL_STRIPPED_DIR,
        BRAIN_MASKS_DIR,
    )
except ImportError:
    NCCT_NIFTI_DIR = None
    NCCT_SKULL_STRIPPED_DIR = None
    BRAIN_MASKS_DIR = None


# ─── Utilities ──────────────────────────────────────────────────────
def run_cmd(cmd):
    """Run command safely and return (ok, stdout, stderr)."""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ok = (p.returncode == 0)
    return ok, p.stdout, p.stderr


# ─── Core function ──────────────────────────────────────────────────
def synthstrip_batch(
    ncct_dir,
    out_stripped_dir=None,
    out_mask_dir=None,
    suffix_stripped="_synthstrip",
    suffix_mask="_brainmask",
    overwrite=False,
    dry_run=False,
):
    ncct_dir = Path(ncct_dir)
    if out_stripped_dir is None:
        out_stripped_dir = ncct_dir.parent / "ncct_synthstrip"
    if out_mask_dir is None:
        out_mask_dir = ncct_dir.parent / "ncct_synthstrip_masks"

    out_stripped_dir = Path(out_stripped_dir)
    out_mask_dir = Path(out_mask_dir)
    out_stripped_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    # find nii and nii.gz
    files = sorted(list(ncct_dir.glob("*.nii")) + list(ncct_dir.glob("*.nii.gz")))
    if not files:
        raise RuntimeError(f"No NIfTI files found in: {ncct_dir}")

    print(f"[INFO] Found {len(files)} NCCT files in: {ncct_dir}")
    print(f"[INFO] Output stripped dir: {out_stripped_dir}")
    print(f"[INFO] Output mask dir:     {out_mask_dir}")

    for i, in_path in enumerate(files, 1):
        in_path = Path(in_path)
        name = in_path.name

        # Keep .nii.gz if present
        if name.endswith(".nii.gz"):
            stem = name[:-7]  # remove .nii.gz
            ext = ".nii.gz"
        else:
            stem = in_path.stem  # removes .nii
            ext = in_path.suffix  # .nii

        out_stripped = out_stripped_dir / f"{stem}{suffix_stripped}{ext}"
        out_mask = out_mask_dir / f"{stem}{suffix_mask}{ext}"

        if not overwrite and out_stripped.exists() and out_mask.exists():
            print(f"[SKIP {i:03d}/{len(files)}] Exists: {in_path.name}")
            continue

        # Build command as a LIST (never shell=True) -> safe with spaces
        cmd_with_mask = [
            "synthstrip-singularity",
            "-i", str(in_path),
            "-o", str(out_stripped),
            "-m", str(out_mask),
        ]

        print(f"[RUN  {i:03d}/{len(files)}] {in_path.name}")
        if dry_run:
            print("  ", " ".join(cmd_with_mask))
            continue

        ok, out, err = run_cmd(cmd_with_mask)

        if ok:
            print("  ✅ done")
            continue

        # If -m not supported by your wrapper, retry without mask flag
        cmd_no_mask = [
            "synthstrip-singularity",
            "-i", str(in_path),
            "-o", str(out_stripped),
        ]

        print("  [WARN] Failed with -m. Retrying without mask output flag...")
        ok2, out2, err2 = run_cmd(cmd_no_mask)

        if ok2:
            print("  ✅ done (no mask flag supported)")
            print("  NOTE: mask not saved because synthstrip-singularity did not accept -m")
        else:
            print("  ❌ FAILED")
            print("  ---- stderr (first try) ----")
            print(err.strip()[:2000])
            print("  ---- stderr (second try) ---")
            print(err2.strip()[:2000])

    print("\n[INFO] Batch complete.")


# ─── CLI ─────────────────────────────────────────────────────────────
def main(cli_args=None):
    ap = argparse.ArgumentParser(
        description="Batch skull-strip NCCT volumes using synthstrip-singularity.",
    )
    ap.add_argument("--ncct-dir", default=NCCT_NIFTI_DIR,
                     help="Directory containing NCCT NIfTI files.")
    ap.add_argument("--out-stripped-dir", default=NCCT_SKULL_STRIPPED_DIR,
                     help="Output directory for skull-stripped images.")
    ap.add_argument("--out-mask-dir", default=BRAIN_MASKS_DIR,
                     help="Output directory for brain masks.")
    ap.add_argument("--overwrite", action="store_true",
                     help="Overwrite existing outputs.")
    ap.add_argument("--dry-run", action="store_true",
                     help="Print commands without executing.")
    args = ap.parse_args(cli_args)

    if args.ncct_dir is None:
        ap.error("--ncct-dir is required (or set NCCT_NIFTI_DIR in config.py)")

    synthstrip_batch(
        ncct_dir=args.ncct_dir,
        out_stripped_dir=args.out_stripped_dir,
        out_mask_dir=args.out_mask_dir,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
