"""
s02_ncct_dicom_to_nifti.py
==========================
Convert an NCCT DICOM series to NIfTI, optionally resampling to 2 mm in Z.

Uses the CSV produced by s00 to locate the correct DICOM folder by
SeriesDescription, picks the series with the most consistent slices,
physically sorts them, and writes a geometry-correct NIfTI via SimpleITK.

Standalone usage
----------------
    python s02_ncct_dicom_to_nifti.py \
        --csv-path dicom_headers.csv \
        --series-desc "NCCT 5mm" \
        --output /out/sub-001_ncct.nii.gz

Pipeline usage
--------------
    from registration.s02_ncct_dicom_to_nifti import (
        dicom_folder_to_nifti_from_csv,
    )
"""

import os
import re
import argparse

import numpy as np
import pandas as pd
import SimpleITK as sitk
import pydicom

try:
    from config import DICOM_HEADERS_CSV, NCCT_NIFTI_DIR
except ImportError:
    DICOM_HEADERS_CSV = None
    NCCT_NIFTI_DIR = None


# ==========================================================
# DICOM helpers
# ==========================================================

def list_files(folder: str):
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
    ]


def dicom_header_quick(fp: str):
    """Read DICOM header only (fast)."""
    ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
    rows = int(getattr(ds, "Rows", -1))
    cols = int(getattr(ds, "Columns", -1))
    sid  = str(getattr(ds, "SeriesInstanceUID", ""))
    iop  = getattr(ds, "ImageOrientationPatient", None)
    ipp  = getattr(ds, "ImagePositionPatient", None)
    inst = int(getattr(ds, "InstanceNumber", 0))
    return rows, cols, sid, iop, ipp, inst


def slice_sort_key(iop, ipp):
    """
    Scalar slice position along normal:
      normal = row_dir x col_dir
      s = dot(IPP, normal)
    """
    row = np.array(iop[:3], dtype=float)
    col = np.array(iop[3:], dtype=float)
    normal = np.cross(row, col)
    nrm = np.linalg.norm(normal)
    if nrm > 0:
        normal = normal / nrm
    pos = np.array(ipp, dtype=float)
    return float(np.dot(pos, normal))


def filter_and_sort_series(folder: str):
    """
    Group by SeriesInstanceUID, keep majority (Rows,Cols), sort by physical position.
    Returns dict:
      out[sid] = {
        "good_sorted": [filepaths...],
        "bad": [filepaths...],
        "major_size": (rows, cols),
        "counts": {(rows, cols): n, ...}
      }
    """
    all_files = list_files(folder)
    per_series = {}

    for fp in all_files:
        try:
            rows, cols, sid, iop, ipp, inst = dicom_header_quick(fp)
            if rows <= 0 or cols <= 0 or sid == "" or iop is None or ipp is None:
                continue
            per_series.setdefault(sid, []).append((fp, rows, cols, iop, ipp, inst))
        except Exception:
            continue

    out = {}
    for sid, items in per_series.items():
        # majority size
        counts = {}
        for _, r, c, *_ in items:
            counts[(r, c)] = counts.get((r, c), 0) + 1
        major_size = max(counts.items(), key=lambda kv: kv[1])[0]

        good = [(fp, iop, ipp, inst) for (fp, r, c, iop, ipp, inst) in items if (r, c) == major_size]
        bad  = [fp for (fp, r, c, *_rest) in items if (r, c) != major_size]

        # sort good by physical slice position
        good_sorted = sorted(good, key=lambda x: (slice_sort_key(x[1], x[2]), x[3]))
        good_sorted = [g[0] for g in good_sorted]

        out[sid] = {
            "good_sorted": good_sorted,
            "bad": bad,
            "major_size": major_size,
            "counts": counts
        }

    return out


# ==========================================================
# Resampling (toggle)
# ==========================================================

def resample_ncct_to_2mm_z(
    ncct: sitk.Image,
    out_spacing=(None, None, 2.0),   # keep x,y; set z=2.0
    interp=sitk.sitkLinear,
    default_value=-1024.0
) -> sitk.Image:
    """
    Resample a 3D NCCT to 2mm slice spacing in Z (keeping X/Y spacing unless specified).
    """
    if ncct.GetDimension() != 3:
        raise ValueError(f"Expected 3D image, got dim={ncct.GetDimension()}")

    in_spacing = ncct.GetSpacing()   # (sx, sy, sz)
    in_size = ncct.GetSize()         # (nx, ny, nz)

    sx = in_spacing[0] if out_spacing[0] is None else float(out_spacing[0])
    sy = in_spacing[1] if out_spacing[1] is None else float(out_spacing[1])
    sz = float(out_spacing[2])

    new_spacing = (sx, sy, sz)

    new_size = [
        int(round(in_size[0] * in_spacing[0] / new_spacing[0])),
        int(round(in_size[1] * in_spacing[1] / new_spacing[1])),
        int(round(in_size[2] * in_spacing[2] / new_spacing[2])),
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ncct)
    resampler.SetInterpolator(interp)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(ncct.GetDirection())
    resampler.SetOutputOrigin(ncct.GetOrigin())
    resampler.SetDefaultPixelValue(float(default_value))
    resampler.SetTransform(sitk.Transform())  # identity

    out = resampler.Execute(sitk.Cast(ncct, sitk.sitkFloat32))
    return sitk.Cast(out, ncct.GetPixelID())


# ==========================================================
# Main: DICOM folder -> NIfTI (+ optional 2mm Z resample)
# ==========================================================

def dicom_folder_to_nifti_from_csv(
    csv_path: str,
    target_series_desc: str,
    out_nii_path: str,
    folder_col: str = "Folder",
    series_col: str = "SeriesDescription",
    compress: bool = True,
    resample_to_2mm: bool = False,
    out_nii_path_2mm: str | None = None,
):
    """
    1) Read CSV
    2) Find rows where SeriesDescription matches target_series_desc
    3) Pick the folder with most files (if NumDicomFiles exists)
    4) In that folder, pick the SeriesInstanceUID with most consistent slices
    5) Load with SITK ImageSeriesReader using physically-sorted filenames
    6) Save NIfTI
    7) (Optional) resample to 2mm in Z and save to out_nii_path_2mm (or overwrite out_nii_path)
    """
    df = pd.read_csv(csv_path)

    matches = df[df[series_col].astype(str) == target_series_desc]
    if matches.empty:
        uniques = df[series_col].dropna().astype(str).unique()
        raise RuntimeError(
            f'No rows where {series_col} == "{target_series_desc}". '
            f"CSV has {len(uniques)} unique series descriptions."
        )

    if "NumDicomFiles" in matches.columns:
        matches = matches.sort_values("NumDicomFiles", ascending=False)

    dcm_dir = matches.iloc[0][folder_col]
    if not isinstance(dcm_dir, str) or not os.path.isdir(dcm_dir):
        raise RuntimeError(f"Folder path invalid: {dcm_dir}")

    series_map = filter_and_sort_series(dcm_dir)
    if not series_map:
        raise RuntimeError(f"No readable DICOM series in: {dcm_dir}")

    # pick series with most good files
    best_id = max(series_map.keys(), key=lambda sid: len(series_map[sid]["good_sorted"]))
    good_files = series_map[best_id]["good_sorted"]
    bad_files = series_map[best_id]["bad"]
    major_size = series_map[best_id]["major_size"]
    counts = series_map[best_id]["counts"]

    print("Picked folder:", dcm_dir)
    print("Picked SeriesID:", best_id)
    print("Size counts:", counts)
    print("Majority (Rows,Cols):", major_size)
    print("GOOD files:", len(good_files))
    print("SKIPPED files:", len(bad_files))

    if len(good_files) == 0:
        raise RuntimeError("No files remain after filtering.")

    # read series
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(good_files)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    img = reader.Execute()

    # save original NIfTI
    os.makedirs(os.path.dirname(out_nii_path), exist_ok=True)
    sitk.WriteImage(img, out_nii_path, useCompression=compress)
    print("Saved (original):", out_nii_path)
    print("  Size:", img.GetSize())
    print("  Spacing:", img.GetSpacing())

    # optional resample
    img_2mm = None
    if resample_to_2mm:
        img_2mm = resample_ncct_to_2mm_z(img, out_spacing=(None, None, 2.0), interp=sitk.sitkLinear, default_value=-1024.0)

        # decide output path
        if out_nii_path_2mm is None:
            # overwrite-style: append suffix next to original
            root, ext = os.path.splitext(out_nii_path)
            if ext == ".gz":
                root2, ext2 = os.path.splitext(root)
                out_nii_path_2mm = root2 + "_z2mm" + ext2 + ext
            else:
                out_nii_path_2mm = root + "_z2mm" + ext

        os.makedirs(os.path.dirname(out_nii_path_2mm), exist_ok=True)
        sitk.WriteImage(img_2mm, out_nii_path_2mm, useCompression=compress)
        print("Saved (resampled z=2mm):", out_nii_path_2mm)
        print("  Size:", img_2mm.GetSize())
        print("  Spacing:", img_2mm.GetSpacing())

    return {
        "image": img,
        "image_2mm": img_2mm,
        "picked_folder": dcm_dir,
        "picked_series_id": best_id,
        "good_files": good_files,
        "bad_files": bad_files,
        "out_path_original": out_nii_path,
        "out_path_2mm": out_nii_path_2mm,
    }


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main(cli_args=None):
    parser = argparse.ArgumentParser(
        description="Convert NCCT DICOM series to NIfTI using a header CSV for lookup."
    )
    parser.add_argument(
        "--patient", "--pid", default=None,
        help="Patient ID (e.g. 01_001). Resolves CSV path and output from config."
    )
    parser.add_argument(
        "--csv-path", default=DICOM_HEADERS_CSV,
        help="Path to the DICOM headers CSV (default: from config)."
    )
    parser.add_argument(
        "--series-desc", default=None,
        help="SeriesDescription string to match in the CSV (required)."
    )
    parser.add_argument(
        "--output", default=None,
        help="Output NIfTI path (overrides config)."
    )
    parser.add_argument(
        "--resample-2mm", action="store_true",
        help="Also resample to 2 mm Z spacing."
    )
    parser.add_argument(
        "--output-2mm", default=None,
        help="Separate output path for the 2 mm resampled NIfTI."
    )
    args = parser.parse_args(cli_args)

    pid = args.patient

    # Resolve CSV
    csv_path = args.csv_path
    if csv_path is None:
        parser.error("--csv-path is required (or set DICOM_HEADERS_CSV in config)")

    # Series description is required
    series_desc = args.series_desc
    if series_desc is None:
        parser.error("--series-desc is required")

    # Resolve output
    output = args.output
    if output is None and pid is not None and NCCT_NIFTI_DIR is not None:
        output = os.path.join(NCCT_NIFTI_DIR, f"sub-{pid}_ses-01_ncct.nii.gz")
    if output is None:
        parser.error("--output is required (or provide --patient with NCCT_NIFTI_DIR in config)")

    dicom_folder_to_nifti_from_csv(
        csv_path=csv_path,
        target_series_desc=series_desc,
        out_nii_path=output,
        resample_to_2mm=args.resample_2mm,
        out_nii_path_2mm=args.output_2mm,
    )


if __name__ == "__main__":
    main()
