"""
Skull-strip registered CTP and DWI using NCCT brain masks.

Refactored from:
    SUS2025-Preprocessing/scripts/skull_stripping/skull_stripping_ctp_dwi.py

For each patient the script:
  1. Loads the NCCT brain mask.
  2. Resamples the mask to the target image geometry (nearest-neighbour).
  3. Optionally erodes the mask to remove thin edge residues.
  4. Multiplies the mask with the DWI (3-D) and/or CTP (4-D) volumes.

Entry point
-----------
    main()                          – batch processing over matched patients
    __main__ argparse block         – CLI wrapper
"""

import os
import re
import argparse
from typing import Dict, List, Optional, Tuple

import SimpleITK as sitk

# ─── Config defaults (optional) ─────────────────────────────────────
try:
    from config import (
        BRAIN_MASKS_DIR,
        DWI_IN_NCCT_DIR,
        CTP_IN_NCCT_DIR,
        DWI_SS_DIR,
        CTP_SS_DIR,
    )
except ImportError:
    BRAIN_MASKS_DIR = None
    DWI_IN_NCCT_DIR = None
    CTP_IN_NCCT_DIR = None
    DWI_SS_DIR = None
    CTP_SS_DIR = None


# ─── Filename patterns ──────────────────────────────────────────────
MASK_PID_REGEXES = [
    r"brain_mask_ncct_(?P<pid>\d{2}_\d{3})\.nii(\.gz)?$",
    r"sub-(?P<pid>\d{2}_\d{3})_.*mask.*\.nii(\.gz)?$",
    r"sub-(?P<pid>\d{2}_\d{3})_ses-01_ncct_brainmask\.nii(\.gz)?$",
]

DWI_PID_REGEX = r"sub-(?P<pid>\d{2}_\d{3})_dwi_in_ncct\.nii(\.gz)?$"

CTP_PID_REGEX = (
    r"sub-(?P<pid>\d{2}_\d{3}).*ctp.*in.*ncct.*\.nii(\.gz)?$"
    r"|sub-(?P<pid2>\d{2}_\d{3})_ses-01_space-ncct_ctp_mc_1fps\.nii(\.gz)?$"
)

SUFFIX = "_ss"

OUTSIDE_DWI = 0.0        # for DWI
OUTSIDE_CTP = -1024.0    # for CT/CTP


# ─── Utilities ──────────────────────────────────────────────────────
def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def is_nifti(fn: str) -> bool:
    return fn.endswith(".nii") or fn.endswith(".nii.gz")


def find_niftis_recursive(root: str) -> List[str]:
    out = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if is_nifti(f):
                out.append(os.path.join(dirpath, f))
    return out


def add_suffix_before_ext(path: str, suffix: str) -> str:
    base = os.path.basename(path)
    if base.endswith(".nii.gz"):
        stem = base[:-7]
        return os.path.join(os.path.dirname(path), stem + suffix + ".nii.gz")
    if base.endswith(".nii"):
        stem = base[:-4]
        return os.path.join(os.path.dirname(path), stem + suffix + ".nii")
    return os.path.join(os.path.dirname(path), base + suffix)


def extract_pid_from_filename(path: str, regex: str) -> Optional[str]:
    m = re.match(regex, os.path.basename(path))
    if not m:
        return None
    if "pid" in m.groupdict() and m.group("pid") is not None:
        return m.group("pid")
    if "pid2" in m.groupdict() and m.group("pid2") is not None:
        return m.group("pid2")
    return None


def extract_pid_from_any_mask_name(path: str) -> Optional[str]:
    name = os.path.basename(path)
    for rgx in MASK_PID_REGEXES:
        m = re.match(rgx, name)
        if m and m.group("pid"):
            return m.group("pid")
    return None


def same_geom(a: sitk.Image, b: sitk.Image) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and a.GetSpacing() == b.GetSpacing()
        and a.GetOrigin() == b.GetOrigin()
        and a.GetDirection() == b.GetDirection()
    )


def erode_mask(mask: sitk.Image, radius: int = 2) -> sitk.Image:
    """Morphological erosion to remove thin edge residues."""
    m = sitk.Cast(mask > 0, sitk.sitkUInt8)
    eroded = sitk.BinaryErode(m, [radius, radius, 0])
    eroded.CopyInformation(mask)
    return eroded


def resample_mask_to_ref(mask3d: sitk.Image, ref3d: sitk.Image) -> sitk.Image:
    m = sitk.Cast(mask3d > 0, sitk.sitkUInt8)
    out = sitk.Resample(
        m,
        ref3d,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    out.CopyInformation(ref3d)
    return out


def apply_mask_3d(
    img3d: sitk.Image,
    mask3d: sitk.Image,
    outside_value: float,
    erode_radius: int = 0,
) -> Tuple[sitk.Image, sitk.Image]:
    m = sitk.Cast(mask3d > 0, sitk.sitkUInt8)
    if not same_geom(img3d, m):
        m = resample_mask_to_ref(m, img3d)
    if erode_radius > 0:
        m = erode_mask(m, erode_radius)
    out = sitk.Mask(img3d, m, outsideValue=float(outside_value))
    out.CopyInformation(img3d)
    return out, m


def apply_mask_4d(
    img4d: sitk.Image,
    mask3d: sitk.Image,
    outside_value: float,
    erode_radius: int = 0,
) -> Tuple[sitk.Image, sitk.Image]:
    if img4d.GetDimension() != 4:
        raise ValueError("Expected 4D image for CTP")

    ref3d = img4d[:, :, :, 0]
    m = sitk.Cast(mask3d > 0, sitk.sitkUInt8)
    if not same_geom(ref3d, m):
        m = resample_mask_to_ref(m, ref3d)
    if erode_radius > 0:
        m = erode_mask(m, erode_radius)

    T = img4d.GetSize()[3]
    vols = []
    for t in range(T):
        v = img4d[:, :, :, t]
        v_ss = sitk.Mask(v, m, outsideValue=float(outside_value))
        v_ss.CopyInformation(v)
        vols.append(v_ss)

    out4d = sitk.JoinSeries(vols)
    out4d.CopyInformation(img4d)
    return out4d, m


# ─── Build lookup maps ──────────────────────────────────────────────
def build_mask_map(mask_dir: str) -> Dict[str, str]:
    masks = {}
    for fp in find_niftis_recursive(mask_dir):
        pid = extract_pid_from_any_mask_name(fp)
        if pid:
            masks[pid] = fp
    return masks


def build_dwi_map(dwi_root: str) -> Dict[str, str]:
    dwis = {}
    for fp in find_niftis_recursive(dwi_root):
        pid = extract_pid_from_filename(fp, DWI_PID_REGEX)
        if pid:
            dwis[pid] = fp
    return dwis


def build_ctp_map(ctp_root: str) -> Dict[str, str]:
    ctps = {}
    for fp in find_niftis_recursive(ctp_root):
        pid = extract_pid_from_filename(fp, CTP_PID_REGEX)
        if pid:
            ctps[pid] = fp
    return ctps


# ─── Main ────────────────────────────────────────────────────────────
def main(
    mask_dir,
    dwi_root,
    ctp_root,
    out_dwi_dir,
    out_ctp_dir,
    save_beside_input=False,
    write_debug_masks=False,
    debug_mask_dir=None,
    pid=None,
    overwrite=False,
    erode_radius=2,
):
    if not save_beside_input:
        ensure_dir(out_dwi_dir)
        ensure_dir(out_ctp_dir)

    if write_debug_masks and debug_mask_dir:
        ensure_dir(debug_mask_dir)

    masks = build_mask_map(mask_dir)
    dwis = build_dwi_map(dwi_root)
    ctps = build_ctp_map(ctp_root)

    if not masks:
        raise RuntimeError(
            f"No masks found in {mask_dir}. "
            "Update MASK_PID_REGEXES to match your mask filenames."
        )

    # choose pids
    if pid:
        pid_list = [pid]
    else:
        pid_list = sorted(set(masks.keys()) & (set(dwis.keys()) | set(ctps.keys())))

    print(f"Found masks: {len(masks)} | DWI: {len(dwis)} | CTP: {len(ctps)}")
    print(f"Processing: {len(pid_list)} subjects")

    for p in pid_list:
        if p not in masks:
            print(f"[SKIP] {p}: mask not found")
            continue

        mask_path = masks[p]
        mask = sitk.ReadImage(mask_path)
        mask = sitk.Cast(mask > 0, sitk.sitkUInt8)

        # ---- DWI ----
        if p in dwis:
            dwi_path = dwis[p]
            out_path = (
                add_suffix_before_ext(dwi_path, SUFFIX)
                if save_beside_input
                else os.path.join(
                    out_dwi_dir,
                    os.path.basename(add_suffix_before_ext(dwi_path, SUFFIX)),
                )
            )

            if (not overwrite) and os.path.exists(out_path):
                print(f"[SKIP] {p} DWI exists: {out_path}")
            else:
                dwi = sitk.ReadImage(dwi_path)
                dwi_ss, m_dwi = apply_mask_3d(dwi, mask, OUTSIDE_DWI, erode_radius=erode_radius)
                sitk.WriteImage(dwi_ss, out_path, useCompression=True)
                print(f"[OK] {p} DWI -> {out_path}")

                if write_debug_masks and debug_mask_dir:
                    dbg = os.path.join(debug_mask_dir, f"sub-{p}_mask_to_dwi.nii.gz")
                    sitk.WriteImage(m_dwi, dbg, useCompression=True)

        # ---- CTP ----
        if p in ctps:
            ctp_path = ctps[p]
            out_path = (
                add_suffix_before_ext(ctp_path, SUFFIX)
                if save_beside_input
                else os.path.join(
                    out_ctp_dir,
                    os.path.basename(add_suffix_before_ext(ctp_path, SUFFIX)),
                )
            )

            if (not overwrite) and os.path.exists(out_path):
                print(f"[SKIP] {p} CTP exists: {out_path}")
            else:
                ctp4d = sitk.ReadImage(ctp_path)
                ctp_ss, m_ctp = apply_mask_4d(ctp4d, mask, OUTSIDE_CTP, erode_radius=erode_radius)
                sitk.WriteImage(ctp_ss, out_path, useCompression=True)
                print(f"[OK] {p} CTP -> {out_path}")

                if write_debug_masks and debug_mask_dir:
                    dbg = os.path.join(debug_mask_dir, f"sub-{p}_mask_to_ctp.nii.gz")
                    sitk.WriteImage(m_ctp, dbg, useCompression=True)

        if p not in dwis and p not in ctps:
            print(f"[WARN] {p}: mask exists but no DWI/CTP found")

    print("✅ Done.")


# ─── CLI ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Skull-strip registered CTP / DWI using NCCT brain masks.",
    )
    ap.add_argument("--mask-dir", default=BRAIN_MASKS_DIR,
                     help="Directory containing NCCT brain masks.")
    ap.add_argument("--dwi-root", default=DWI_IN_NCCT_DIR,
                     help="Root directory for DWI NIfTIs.")
    ap.add_argument("--ctp-root", default=CTP_IN_NCCT_DIR,
                     help="Root directory for CTP NIfTIs in NCCT space.")
    ap.add_argument("--out-dwi-dir", default=DWI_SS_DIR,
                     help="Output directory for skull-stripped DWI.")
    ap.add_argument("--out-ctp-dir", default=CTP_SS_DIR,
                     help="Output directory for skull-stripped CTP.")
    ap.add_argument("--save-beside-input", action="store_true",
                     help="Save outputs beside input files instead of dedicated dirs.")
    ap.add_argument("--write-debug-masks", action="store_true",
                     help="Save resampled masks for debug inspection.")
    ap.add_argument("--debug-mask-dir", default=None,
                     help="Directory for debug mask outputs.")
    ap.add_argument("--pid", default=None,
                     help="Process only one patient id, e.g. 01_002.")
    ap.add_argument("--overwrite", action="store_true",
                     help="Overwrite existing outputs.")
    ap.add_argument("--erode", type=int, default=2,
                     help="Erosion radius (voxels) to remove edge residues. 0=off (default: 2).")
    args = ap.parse_args()

    if args.mask_dir is None:
        ap.error("--mask-dir is required (or set BRAIN_MASKS_DIR in config.py)")
    if args.dwi_root is None and args.ctp_root is None:
        ap.error("At least one of --dwi-root / --ctp-root is required "
                 "(or set DWI_IN_NCCT_DIR / CTP_IN_NCCT_DIR in config.py)")

    main(
        mask_dir=args.mask_dir,
        dwi_root=args.dwi_root or "",
        ctp_root=args.ctp_root or "",
        out_dwi_dir=args.out_dwi_dir or "",
        out_ctp_dir=args.out_ctp_dir or "",
        save_beside_input=args.save_beside_input,
        write_debug_masks=args.write_debug_masks,
        debug_mask_dir=args.debug_mask_dir,
        pid=args.pid,
        overwrite=args.overwrite,
        erode_radius=args.erode,
    )
