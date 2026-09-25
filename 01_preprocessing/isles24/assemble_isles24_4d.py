#!/usr/bin/env python3
"""assemble_isles24_4d.py

Reassemble the ISLES24 nnUNet-format channel files (one 3D volume per time
point, 40 channels per case) into one 4D CTP NIfTI per patient, matching the
SUS curated format:

  output_root/sub-strokeXXXX/ses-01/sub-strokeXXXX_ses-01_space-ncct_ctp.nii.gz

The 4D volume has SimpleITK size (X, Y, Z, T=40) and corresponding numpy shape
(40, Z, Y, X) when read via sitk.GetArrayFromImage. Identical convention to
the SUS curated CTPs, so the baseline/GLCM/mJ-Net/nnUNet feature extraction
scripts can be re-used unchanged.
"""

import argparse
import os
import sys
import glob
import re
from pathlib import Path

import numpy as np
import SimpleITK as sitk


# ---------------- defaults ----------------
DEFAULT_INPUT_DIR  = os.path.join(
    os.environ.get("NNUNET_DATASET_ROOT", "/path/to/nnUNetFrame/dataset"),
    "nnUNet_raw", "Dataset1155_ISLES24_CTP_3DMultiChannel", "imagesTs")
DEFAULT_OUTPUT_DIR = os.path.join(
    os.environ.get("ISLES24_WORK_ROOT", "/path/to/isles24-work"),
    "isles24_curated", "derivatives")

N_CHANNELS = 40
TIME_SPACING = 1.0  # seconds — ISLES24 CTPs are typically 1 s spacing
TIME_ORIGIN  = 0.0


# ---------------- helpers ----------------
def case_to_pid(case_id: str) -> str:
    """case_0173 -> sub-stroke0173"""
    m = re.fullmatch(r"case_(\d+)", case_id)
    if not m:
        raise ValueError(f"Unexpected case id: {case_id}")
    return f"sub-stroke{m.group(1)}"


def discover_cases(input_dir: Path):
    files = sorted(input_dir.glob("case_*_*.nii.gz"))
    cases = {}
    for f in files:
        m = re.fullmatch(r"(case_\d+)_(\d+)\.nii\.gz", f.name)
        if not m:
            continue
        cid, t = m.group(1), int(m.group(2))
        cases.setdefault(cid, {})[t] = f
    return cases


def assemble_4d(channel_files):
    """
    channel_files : dict {t: Path} of 40 channels.
    Returns a 4D SimpleITK image with the metadata of the first channel
    extended along a time axis.
    """
    ts = sorted(channel_files.keys())
    if ts != list(range(N_CHANNELS)):
        raise RuntimeError(f"Missing/extra channels — found {ts}, expected 0..{N_CHANNELS-1}")

    # Read first channel to grab metadata
    img0 = sitk.ReadImage(str(channel_files[0]))
    spacing3 = img0.GetSpacing()        # (X, Y, Z)
    origin3  = img0.GetOrigin()         # (X, Y, Z)
    dir3     = img0.GetDirection()      # 9-tuple

    # Stack: each channel array is (Z, Y, X); stack along axis 0 -> (T, Z, Y, X)
    # np.flipud per 3D volume = flip Z axis (axis 0 of (Z, Y, X))
    arrs = []
    for t in ts:
        img_t = sitk.ReadImage(str(channel_files[t]))
        if img_t.GetSize() != img0.GetSize():
            raise RuntimeError(
                f"Channel {t} size {img_t.GetSize()} != channel 0 size {img0.GetSize()}"
            )
        arr_t = sitk.GetArrayFromImage(img_t)
        # arr_t = np.flipud(arr_t)
        arrs.append(arr_t)
    arr_4d = np.stack(arrs, axis=0).astype(np.float32)         # (T, Z, Y, X)

    # Build 4D SimpleITK image. Numpy (T, Z, Y, X) -> SITK size (X, Y, Z, T)
    img4 = sitk.GetImageFromArray(arr_4d, isVector=False)
    img4.SetSpacing(spacing3 + (TIME_SPACING,))
    img4.SetOrigin(origin3 + (TIME_ORIGIN,))

    # Extend 3x3 direction to 4x4 (identity for the time axis)
    d = list(dir3)
    d4 = [d[0], d[1], d[2], 0.0,
          d[3], d[4], d[5], 0.0,
          d[6], d[7], d[8], 0.0,
          0.0,  0.0,  0.0,  1.0]
    img4.SetDirection(tuple(d4))

    return img4


def write_for_patient(pid: str, img4: sitk.Image, output_root: Path) -> Path:
    """Write at output_root/<pid>/ses-01/<pid>_ses-01_space-ncct_ctp.nii.gz"""
    out_dir = output_root / pid / "ses-01"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pid}_ses-01_space-ncct_ctp.nii.gz"
    sitk.WriteImage(img4, str(out_path))
    return out_path


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default=DEFAULT_INPUT_DIR,
                    help="Folder containing case_XXXX_TTTT.nii.gz channel files.")
    ap.add_argument("--output", default=DEFAULT_OUTPUT_DIR,
                    help="Output root (BIDS-like: <pid>/ses-01/<pid>_ses-01_space-ncct_ctp.nii.gz).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing outputs.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N cases (for testing).")
    args = ap.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = discover_cases(input_dir)
    case_ids = sorted(cases.keys())
    if args.limit:
        case_ids = case_ids[:args.limit]
    print(f"Found {len(case_ids)} cases in {input_dir}", flush=True)
    print(f"Output -> {output_dir}", flush=True)

    failed = []
    skipped = []
    written = []
    for i, cid in enumerate(case_ids, 1):
        pid = case_to_pid(cid)
        out_path = output_dir / pid / "ses-01" / f"{pid}_ses-01_space-ncct_ctp.nii.gz"

        if out_path.exists() and not args.overwrite:
            print(f"[{i}/{len(case_ids)}] {cid} -> {pid}  (skip; exists)", flush=True)
            skipped.append(cid)
            continue

        try:
            img4 = assemble_4d(cases[cid])
            written_path = write_for_patient(pid, img4, output_dir)
            arr_shape = sitk.GetArrayFromImage(img4).shape
            print(f"[{i}/{len(case_ids)}] {cid} -> {pid}  shape={arr_shape}  -> {written_path}",
                  flush=True)
            written.append(cid)
        except Exception as e:
            print(f"[{i}/{len(case_ids)}] {cid} FAILED: {e}", flush=True)
            failed.append((cid, str(e)))

    print(f"\nDone. written={len(written)}  skipped={len(skipped)}  failed={len(failed)}")
    if failed:
        print("\nFailed cases:")
        for cid, msg in failed:
            print(f"  {cid}: {msg}")


if __name__ == "__main__":
    main()
