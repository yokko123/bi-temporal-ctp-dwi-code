"""
Skull-strip 4-D CTP (in NCCT space) using native CTP masks + CTP→NCCT transforms.

Refactored from:
    SUS2025-Preprocessing/scripts/skull_stripping/skull_strip_ctp_with_native_masks.py

Pipeline per patient:
  1. Stack 2-D PNG brain masks (from pre-registration CTP) into a 3-D volume.
  2. Attach the original CTP geometry (from the motion-corrected NIfTI).
  3. Apply the CTP→NCCT rigid transform to bring the mask into NCCT space
     (using the 3-D CTP reference in NCCT as the target geometry).
  4. Multiply every timepoint of the 4-D CTP-in-NCCT by the transformed mask.

Entry point
-----------
    main()                          – batch processing over matched patients
    __main__ argparse block         – CLI wrapper
"""

import argparse
import os
import re
from typing import Dict, Optional

import numpy as np
import SimpleITK as sitk

# ─── Config defaults (optional) ─────────────────────────────────────
try:
    from config import (
        CTP_NATIVE_MASKS,
        CTP_MC_DIR,
        TRANSFORMS_DIR,
        CTP_3D_REF_DIR,
        CTP_IN_NCCT_DIR,
        CTP_SS_NATIVE_DIR,
        CTP_SS_BRAINMASKS_DIR,
    )
except ImportError:
    CTP_NATIVE_MASKS = None
    CTP_MC_DIR = None
    TRANSFORMS_DIR = None
    CTP_3D_REF_DIR = None
    CTP_IN_NCCT_DIR = None
    CTP_SS_NATIVE_DIR = None
    CTP_SS_BRAINMASKS_DIR = None


# ─── Helpers ─────────────────────────────────────────────────────────
def extract_pid(filename: str) -> Optional[str]:
    """Extract patient id like '01_001' from various naming patterns."""
    # sub-01_001_ses-01_...
    m = re.search(r"sub-(\d{2}_\d{3})", filename)
    if m:
        return m.group(1)
    # CTP_01_001
    m = re.search(r"CTP_(\d{2}_\d{3})", filename)
    if m:
        return m.group(1)
    return None


def load_png_mask_stack(mask_folder: str, ref_sitk: sitk.Image) -> sitk.Image:
    """
    Load numbered PNG slices from a folder and stack into a 3-D SimpleITK
    image with geometry taken from ref_sitk (the original CTP volume).

    PNG slices are named 01.png, 02.png, ... and are boolean/binary.
    They are ordered by filename (ascending = bottom→top in NIfTI z).
    """
    pngs = sorted(
        [f for f in os.listdir(mask_folder) if f.lower().endswith(".png")]
    )
    if not pngs:
        raise FileNotFoundError(f"No PNG masks in {mask_folder}")

    slices = []
    for fn in pngs:
        s = sitk.ReadImage(os.path.join(mask_folder, fn), sitk.sitkUInt8)
        arr = sitk.GetArrayFromImage(s)  # (H, W) or (H, W, C)
        if arr.ndim == 3:
            arr = arr[..., 0]
        slices.append((arr > 0).astype(np.uint8))

    vol = np.stack(slices, axis=0)  # (Z, H, W)
    mask3d = sitk.GetImageFromArray(vol)

    # Copy spatial metadata from the 3-D geometry of the source CTP
    # ref_sitk can be 4-D; extract 3-D info
    if ref_sitk.GetDimension() == 4:
        spacing = ref_sitk.GetSpacing()[:3]
        origin = ref_sitk.GetOrigin()[:3]
        direction = ref_sitk.GetDirection()
        # 4-D direction is 16 elements; extract 3×3 sub-matrix
        dir3 = [direction[i * 4 + j] for i in range(3) for j in range(3)]
    else:
        spacing = ref_sitk.GetSpacing()
        origin = ref_sitk.GetOrigin()
        dir3 = list(ref_sitk.GetDirection())

    mask3d.SetSpacing(spacing)
    mask3d.SetOrigin(origin)
    mask3d.SetDirection(dir3)
    return mask3d


def transform_mask_to_ncct(
    mask_ctp: sitk.Image,
    tfm_path: str,
    ref_ncct: sitk.Image,
) -> sitk.Image:
    """
    Apply the CTP→NCCT transform to bring the mask from CTP space
    into NCCT space, resampled onto the ref_ncct grid.
    """
    tfm = sitk.ReadTransform(tfm_path)
    resampled = sitk.Resample(
        mask_ctp,
        ref_ncct,
        tfm,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    resampled.CopyInformation(ref_ncct)
    return resampled


def erode_mask(mask: sitk.Image, radius: int) -> sitk.Image:
    """Binary erosion to trim edge residues."""
    eroded = sitk.BinaryErode(mask, [radius] * 3)
    eroded.CopyInformation(mask)
    return eroded


def apply_mask_4d(
    img4d: sitk.Image,
    mask3d: sitk.Image,
    outside_value: float = 0.0,
) -> sitk.Image:
    """Apply a 3-D mask to every timepoint of a 4-D image."""
    T = img4d.GetSize()[3]
    vols = []
    for t in range(T):
        v = img4d[:, :, :, t]
        v_masked = sitk.Mask(v, mask3d, outsideValue=float(outside_value))
        v_masked.CopyInformation(v)
        vols.append(v_masked)
    out = sitk.JoinSeries(vols)
    out.CopyInformation(img4d)
    return out


# ─── Build lookup maps ──────────────────────────────────────────────
def build_map(directory: str, pattern: str, glob_ext: str = "") -> Dict[str, str]:
    """Build pid→path dict from files or folders in directory."""
    result = {}
    if not os.path.isdir(directory):
        return result
    for entry in os.listdir(directory):
        pid = extract_pid(entry)
        if pid is None:
            continue
        full = os.path.join(directory, entry)
        if glob_ext and not entry.endswith(glob_ext):
            continue
        result[pid] = full
    return result


# ─── Main ────────────────────────────────────────────────────────────
def main(
    mask_dir,
    ctp_mc_dir,
    transform_dir,
    ref_dir,
    ctp_ncct_dir,
    out_dir,
    mask_out_dir=None,
    erode_radius=0,
    pid=None,
    overwrite=False,
    dry_run=False,
):
    os.makedirs(out_dir, exist_ok=True)
    if mask_out_dir:
        os.makedirs(mask_out_dir, exist_ok=True)

    # Build lookup tables
    masks = build_map(mask_dir, "CTP_")           # CTP_01_001 folders
    ctp_mc = build_map(ctp_mc_dir, "sub-", ".nii.gz")
    transforms = build_map(transform_dir, "sub-", ".tfm")
    refs = build_map(ref_dir, "sub-", ".nii.gz")
    ctp_ncct = build_map(ctp_ncct_dir, "sub-", ".nii.gz")

    print(f"[INFO] Masks: {len(masks)} | MC-CTP: {len(ctp_mc)} | "
          f"Transforms: {len(transforms)} | Refs: {len(refs)} | CTP-NCCT: {len(ctp_ncct)}")

    # Determine patient list
    if pid:
        pid_list = [pid]
    else:
        pid_list = sorted(
            set(masks) & set(ctp_mc) & set(transforms) & set(refs) & set(ctp_ncct)
        )

    print(f"[INFO] Processing {len(pid_list)} patients (erode={erode_radius})\n")

    for i, p in enumerate(pid_list, 1):
        # Check all required files exist for this patient
        missing = []
        if p not in masks:
            missing.append("mask")
        if p not in ctp_mc:
            missing.append("ctp_mc")
        if p not in transforms:
            missing.append("transform")
        if p not in refs:
            missing.append("ref")
        if p not in ctp_ncct:
            missing.append("ctp_ncct")
        if missing:
            print(f"[MISS {i:03d}] {p}: missing {', '.join(missing)}")
            continue

        # Output path
        ctp_ncct_name = os.path.basename(ctp_ncct[p])
        if ctp_ncct_name.endswith(".nii.gz"):
            stem = ctp_ncct_name[:-7]
            ext = ".nii.gz"
        else:
            stem = ctp_ncct_name[:-4]
            ext = ".nii"
        out_path = os.path.join(out_dir, f"{stem}_ss{ext}")

        if not overwrite and os.path.exists(out_path):
            print(f"[SKIP {i:03d}] {p}")
            continue

        print(f"[RUN  {i:03d}/{len(pid_list)}] {p}")

        if dry_run:
            print(f"  mask:  {masks[p]}")
            print(f"  mc:    {os.path.basename(ctp_mc[p])}")
            print(f"  tfm:   {os.path.basename(transforms[p])}")
            print(f"  ref:   {os.path.basename(refs[p])}")
            print(f"  input: {ctp_ncct_name}")
            print(f"  out:   {os.path.basename(out_path)}")
            continue

        # 1. Load PNG mask slices + original CTP geometry
        ctp_mc_img = sitk.ReadImage(ctp_mc[p])
        mask_ctp = load_png_mask_stack(masks[p], ctp_mc_img)

        # 2. Transform mask CTP→NCCT
        ref_img = sitk.ReadImage(refs[p])
        mask_ncct = transform_mask_to_ncct(mask_ctp, transforms[p], ref_img)

        # 3. Optional erosion
        if erode_radius > 0:
            mask_ncct = erode_mask(mask_ncct, erode_radius)

        # 4. Apply mask to 4D CTP-in-NCCT
        ctp4d = sitk.ReadImage(ctp_ncct[p])
        ctp_ss = apply_mask_4d(ctp4d, mask_ncct, outside_value=0.0)

        # 5. Save
        sitk.WriteImage(ctp_ss, out_path, useCompression=True)
        print(f"  -> {out_path}")

        if mask_out_dir:
            mask_save = os.path.join(
                mask_out_dir, f"sub-{p}_ses-01_space-ncct_brainmask{ext}"
            )
            sitk.WriteImage(mask_ncct, mask_save, useCompression=True)

    print("\n[INFO] Done.")


# ─── CLI ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Skull-strip CTP-in-NCCT using native CTP masks + CTP→NCCT transforms.",
    )
    ap.add_argument("--mask-dir", default=CTP_NATIVE_MASKS,
                     help="Directory with native CTP mask folders (PNG slices).")
    ap.add_argument("--ctp-mc-dir", default=CTP_MC_DIR,
                     help="Dir with motion-corrected CTP NIfTIs (for geometry).")
    ap.add_argument("--transform-dir", default=TRANSFORMS_DIR,
                     help="Dir with CTP→NCCT .tfm transforms.")
    ap.add_argument("--ref-dir", default=CTP_3D_REF_DIR,
                     help="Dir with 3-D CTP ref volumes in NCCT space.")
    ap.add_argument("--ctp-ncct-dir", default=CTP_IN_NCCT_DIR,
                     help="Dir with 4-D CTP already registered to NCCT space.")
    ap.add_argument("--out-dir", default=CTP_SS_NATIVE_DIR,
                     help="Output directory for skull-stripped CTP.")
    ap.add_argument("--mask-out-dir", default=CTP_SS_BRAINMASKS_DIR,
                     help="Save transformed brain masks (optional).")
    ap.add_argument("--erode", type=int, default=0,
                     help="Erosion radius (voxels) applied after transform (default: 0).")
    ap.add_argument("--pid", default=None,
                     help="Process single patient, e.g. 01_001.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                     help="Preview matching without processing.")
    args = ap.parse_args()

    if args.mask_dir is None:
        ap.error("--mask-dir is required (or set CTP_NATIVE_MASKS in config.py)")
    if args.out_dir is None:
        ap.error("--out-dir is required (or set CTP_SS_NATIVE_DIR in config.py)")

    main(
        mask_dir=args.mask_dir,
        ctp_mc_dir=args.ctp_mc_dir or "",
        transform_dir=args.transform_dir or "",
        ref_dir=args.ref_dir or "",
        ctp_ncct_dir=args.ctp_ncct_dir or "",
        out_dir=args.out_dir,
        mask_out_dir=args.mask_out_dir,
        erode_radius=args.erode,
        pid=args.pid,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
