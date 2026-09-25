"""
s04_dwi_dicom_to_nifti.py
=========================
Convert DWI DICOM series to NIfTI.

Locates the correct DICOM folder via a CSV (produced by s00), filters slices
by majority matrix size, optionally selects a b-value group (e.g. high-b for
"dark" DWI), physically sorts them along the slice normal, and writes a
geometry-correct NIfTI via SimpleITK.

Standalone usage
----------------
    python s04_dwi_dicom_to_nifti.py \\
        --csv-path dicom_headers.csv \\
        --series-desc "ep2d_diff_4scan_trace_p3_TRACEW" \\
        --output /out/sub-001_dwi.nii.gz

Pipeline usage
--------------
    from registration.s04_dwi_dicom_to_nifti import (
        dicom_folder_to_nifti_dwi_from_csv_with_skip_and_plot,
    )
"""

import os
import math
import argparse

import numpy as np
import pandas as pd
import SimpleITK as sitk
import pydicom
import matplotlib.pyplot as plt

try:
    from config import DICOM_HEADERS_CSV, DWI_NIFTI_DIR
except ImportError:
    DICOM_HEADERS_CSV = None
    DWI_NIFTI_DIR = None


# ==========================================================
# VIS
# ==========================================================
def montage_slices(vol_zyx, title="", max_cols=12, figsize=(18, 18), vmin=None, vmax=None):
    Z = vol_zyx.shape[0]
    cols = min(max_cols, Z) if Z > 0 else 1
    rows = math.ceil(Z / cols) if Z > 0 else 1

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < Z:
            ax.imshow(vol_zyx[i], vmin=vmin, vmax=vmax, cmap="gray")
            ax.set_title(f"z={i}", fontsize=8)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


# ==========================================================
# DICOM HELPERS
# ==========================================================
def list_files(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f))]


def _safe_float(x):
    try:
        if hasattr(x, "value"):
            x = x.value
        if isinstance(x, (list, tuple)) and len(x) > 0:
            x = x[0]
        return float(x)
    except Exception:
        return None


def get_bval(ds):
    """
    Try to read diffusion b-value from common tags.
    Returns float or None.
    """
    # Standard DICOM (Enhanced MR) diffusion b-value
    if (0x0018, 0x9087) in ds:
        return _safe_float(ds[(0x0018, 0x9087)])

    # Siemens private (very common)
    if (0x0019, 0x100C) in ds:
        return _safe_float(ds[(0x0019, 0x100C)])

    # GE private (sometimes)
    if (0x0043, 0x1039) in ds:
        v = ds[(0x0043, 0x1039)].value
        # often a list; first element may encode b-value-ish
        try:
            if isinstance(v, (list, tuple)) and len(v) > 0:
                return float(v[0])
        except Exception:
            pass

    return None


def dicom_header_quick(fp):
    """Read header only."""
    ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)

    rows = int(getattr(ds, "Rows", -1))
    cols = int(getattr(ds, "Columns", -1))
    sid  = str(getattr(ds, "SeriesInstanceUID", ""))
    desc = str(getattr(ds, "SeriesDescription", "")).strip()

    iop = getattr(ds, "ImageOrientationPatient", None)
    ipp = getattr(ds, "ImagePositionPatient", None)
    inst = int(getattr(ds, "InstanceNumber", 0))

    bval = get_bval(ds)

    return rows, cols, sid, desc, iop, ipp, inst, bval


def slice_sort_key(iop, ipp):
    row = np.array(iop[:3], dtype=float)
    col = np.array(iop[3:], dtype=float)
    normal = np.cross(row, col)
    nrm = np.linalg.norm(normal)
    if nrm > 0:
        normal = normal / nrm
    pos = np.array(ipp, dtype=float)
    return float(np.dot(pos, normal))


def _choose_bval_group(items, prefer="max_nonzero", bval_min=500.0):
    """
    items: list of tuples (fp, rows, cols, iop, ipp, inst, bval)

    prefer:
      - "max_nonzero": choose group with highest bval (>= bval_min if possible)
      - "min"       : choose lowest bval group (usually b0)
      - "any"       : don't filter
    """
    if prefer == "any":
        return items, None

    # Group by rounded bval (because some tags are 999.999 etc.)
    groups = {}
    for it in items:
        b = it[-1]
        if b is None:
            key = None
        else:
            key = int(round(float(b)))
        groups.setdefault(key, []).append(it)

    # If no bvals at all, fall back to "no filter"
    if set(groups.keys()) == {None}:
        return items, None

    keys = [k for k in groups.keys() if k is not None]
    if not keys:
        return items, None

    if prefer == "min":
        chosen_key = min(keys)
        return groups[chosen_key], chosen_key

    # default: max_nonzero
    # Prefer b-values >= bval_min; if none, just take max
    keys_ge = [k for k in keys if k >= int(bval_min)]
    chosen_key = max(keys_ge) if keys_ge else max(keys)
    return groups[chosen_key], chosen_key


def filter_and_sort_series(folder, bval_prefer="max_nonzero", bval_min=500.0):
    """
    Returns dict per series_id:
      - good_sorted: selected files (optionally filtered by b-value), sorted by physical slice position
      - bad: skipped files (other sizes)
      - major_size, counts
      - series_desc
      - chosen_bval
      - bval_counts
    """
    all_files = list_files(folder)
    per_series = {}

    for fp in all_files:
        try:
            rows, cols, sid, desc, iop, ipp, inst, bval = dicom_header_quick(fp)
            if rows <= 0 or cols <= 0 or sid == "" or iop is None or ipp is None:
                continue
            per_series.setdefault(sid, {"desc": desc, "items": []}).get("items").append(
                (fp, rows, cols, iop, ipp, inst, bval)
            )
        except Exception:
            continue

    out = {}
    for sid, pack in per_series.items():
        desc  = pack["desc"]
        items = pack["items"]

        # majority (Rows,Cols)
        counts = {}
        for _, r, c, *_ in items:
            counts[(r, c)] = counts.get((r, c), 0) + 1
        major_size = max(counts.items(), key=lambda kv: kv[1])[0]

        good = [(fp, iop, ipp, inst, bval)
                for (fp, r, c, iop, ipp, inst, bval) in items if (r, c) == major_size]
        bad  = [fp for (fp, r, c, *_rest) in items if (r, c) != major_size]

        # bval filtering happens on the good list
        # Convert back to a structure _choose_bval_group expects
        good_full = [(fp, major_size[0], major_size[1], iop, ipp, inst, bval)
                     for (fp, iop, ipp, inst, bval) in good]

        # track bval counts for debugging
        bval_counts = {}
        for it in good_full:
            b = it[-1]
            key = None if b is None else int(round(float(b)))
            bval_counts[key] = bval_counts.get(key, 0) + 1

        chosen_items, chosen_bval = _choose_bval_group(
            good_full, prefer=bval_prefer, bval_min=bval_min
        )

        # sort by physical slice position
        chosen_sorted = sorted(chosen_items, key=lambda x: (slice_sort_key(x[3], x[4]), x[5]))
        chosen_sorted = [c[0] for c in chosen_sorted]  # filenames only

        out[sid] = {
            "good_sorted": chosen_sorted,
            "bad": bad,
            "major_size": major_size,
            "counts": counts,
            "series_desc": desc,
            "chosen_bval": chosen_bval,
            "bval_counts": bval_counts,
        }

    return out


# ==========================================================
# DEBUG
# ==========================================================
def debug_selected_spacing(good_files):
    d0 = pydicom.dcmread(good_files[0], stop_before_pixels=True, force=True)
    iop = np.array(d0.ImageOrientationPatient, dtype=float)
    row = iop[:3]
    col = iop[3:]
    normal = np.cross(row, col)
    normal = normal / (np.linalg.norm(normal) + 1e-8)

    s_positions = []
    for fp in good_files:
        ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
        ipp = np.array(ds.ImagePositionPatient, dtype=float)
        s_positions.append(float(np.dot(ipp, normal)))

    s_positions = np.array(s_positions)
    diffs = np.diff(np.sort(s_positions))

    print("Selected slices:", len(good_files))
    print("z-step stats (mm): min/median/max =",
          float(np.min(diffs)), float(np.median(diffs)), float(np.max(diffs)))
    print("Unique-ish steps:", np.unique(np.round(diffs, 3))[:20])


# ==========================================================
# CORE: DICOM folder -> NIfTI by SeriesDescription + choose "dark DWI" via bval
# ==========================================================
def dicom_folder_to_nifti_dwi_from_csv_with_skip_and_plot(
    csv_path: str,
    target_series_desc: str,
    out_nii_path: str,
    folder_col: str = "Folder",
    series_col: str = "SeriesDescription",
    compress: bool = True,
    montage_cols: int = 10,
    plot: bool = True,
    # NEW: pick darker DWI using bval
    bval_prefer: str = "max_nonzero",   # keep "dark DWI" (high b-value)
    bval_min: float = 500.0,            # typical threshold to avoid b0
):
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

    series_map = filter_and_sort_series(dcm_dir, bval_prefer=bval_prefer, bval_min=bval_min)
    if not series_map:
        raise RuntimeError(f"No readable DICOM series in: {dcm_dir}")

    # choose the SeriesInstanceUID whose DICOM SeriesDescription matches
    candidates = []
    for sid, info in series_map.items():
        if str(info.get("series_desc", "")).strip() == str(target_series_desc).strip():
            if len(info["good_sorted"]) > 0:
                candidates.append((sid, info))

    if not candidates:
        print("\n[DEBUG] Available series in folder:")
        for sid, info in sorted(series_map.items(), key=lambda kv: len(kv[1]["good_sorted"]), reverse=True):
            print(f"  n={len(info['good_sorted']):4d} | desc='{info.get('series_desc','')}' | sid={sid} | bvals={info.get('bval_counts')}")
        raise RuntimeError(f'No series matched SeriesDescription="{target_series_desc}"')

    best_id, best_info = max(candidates, key=lambda x: len(x[1]["good_sorted"]))
    good_files  = best_info["good_sorted"]
    bad_files   = best_info["bad"]
    major_size  = best_info["major_size"]
    counts      = best_info["counts"]
    bval_counts = best_info["bval_counts"]
    chosen_bval = best_info["chosen_bval"]

    print("\n==============================")
    print("Target SeriesDescription:", target_series_desc)
    print("Picked folder:", dcm_dir)
    print("Picked SeriesID:", best_id)
    print("Majority (Rows,Cols):", major_size)
    print("GOOD files:", len(good_files))
    print("SKIPPED files:", len(bad_files))
    print("b-value counts:", bval_counts)
    print("Chosen b-value group:", chosen_bval, f"(prefer={bval_prefer}, bval_min={bval_min})")

    debug_selected_spacing(good_files)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(good_files)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    img = reader.Execute()
    print("SITK spacing:", img.GetSpacing())

    os.makedirs(os.path.dirname(out_nii_path), exist_ok=True)
    sitk.WriteImage(img, out_nii_path, useCompression=compress)

    print("Saved:", out_nii_path)
    print("Size:", img.GetSize())
    print("Spacing:", img.GetSpacing())
    print("Origin:", img.GetOrigin())
    print("Direction:", img.GetDirection())

    if plot:
        arr_good = sitk.GetArrayFromImage(img)  # (Z,Y,X)
        montage_slices(
            arr_good,
            title=f"DWI (b={chosen_bval}) | {target_series_desc} | Z={arr_good.shape[0]}",
            max_cols=montage_cols,
            figsize=(15, 15),
        )

    return img, dcm_dir, best_id, good_files, bad_files


# ==========================================================
# Auto-discover DWI series from CSV
# ==========================================================
def _auto_discover_dwi_series(csv_path, pid):
    """Try to find a DWI SeriesDescription for *pid* in the DICOM headers CSV.

    Heuristic: look for MR-modality rows whose folder path contains the
    patient ID and whose SeriesDescription contains 'DWI' (case-insensitive).
    If exactly one unique description is found, return it; otherwise print
    candidates and return None so the caller can raise a proper error.
    """
    try:
        import pandas as pd
    except ImportError:
        return None
    if not os.path.isfile(csv_path):
        return None

    df = pd.read_csv(csv_path)

    # Search both underscore and hyphen variants of the patient ID
    pid_variants = {pid, pid.replace("_", "-"), pid.replace("-", "_")}
    mask = pd.Series(False, index=df.index)
    for v in pid_variants:
        mask |= df["Folder"].astype(str).str.contains(v, na=False)

    mr = df[mask & (df["Modality"].astype(str).str.strip().str.upper() == "MR")]
    dwi = mr[mr["SeriesDescription"].astype(str).str.contains("DWI", case=False, na=False)]
    descs = dwi["SeriesDescription"].dropna().unique().tolist()

    if len(descs) == 1:
        print(f"[s04] Auto-discovered DWI series for {pid}: '{descs[0]}'")
        return descs[0]

    if len(descs) > 1:
        print(f"[s04] Multiple DWI series found for {pid}:")
        for d in descs:
            print(f"       - {d}")
        print("       Please specify one with --series-desc")
    else:
        # No DWI series found — show all MR series as hints
        all_mr_descs = mr["SeriesDescription"].dropna().unique().tolist()
        if all_mr_descs:
            print(f"[s04] No DWI series auto-discovered for {pid}. Available MR series:")
            for d in all_mr_descs:
                print(f"       - {d}")
            print("       Please specify one with --series-desc")

    return None


# ==========================================================
# CLI
# ==========================================================
def main(cli_args=None):
    parser = argparse.ArgumentParser(
        description="Convert DWI DICOM series to NIfTI.",
    )
    parser.add_argument("--patient", "--pid", default=None,
                        help="Patient ID (e.g. 01_001). Resolves CSV path and output from config.")
    parser.add_argument("--csv-path", default=DICOM_HEADERS_CSV,
                        help="CSV with DICOM header info (default: from config).")
    parser.add_argument("--series-desc", default=None,
                        help="Exact SeriesDescription to match in the CSV (required).")
    parser.add_argument("--output", default=None,
                        help="Output NIfTI path (overrides config).")
    parser.add_argument("--bval-prefer", default="max_nonzero",
                        choices=["max_nonzero", "min", "any"],
                        help="B-value group selection strategy (default: max_nonzero).")
    parser.add_argument("--bval-min", type=float, default=500.0,
                        help="Minimum b-value threshold for max_nonzero (default: 500).")
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable montage plot.")
    args = parser.parse_args(cli_args)

    pid = args.patient

    # Resolve CSV
    csv_path = args.csv_path
    if csv_path is None:
        parser.error("--csv-path is required (or set DICOM_HEADERS_CSV in config)")

    # Series description — required, but can be auto-discovered from CSV
    series_desc = args.series_desc
    if series_desc is None and pid is not None and csv_path is not None:
        series_desc = _auto_discover_dwi_series(csv_path, pid)
    if series_desc is None:
        parser.error("--series-desc is required (could not auto-discover DWI series)")

    # Resolve output
    output = args.output
    if output is None and pid is not None and DWI_NIFTI_DIR is not None:
        out_dir = os.path.join(DWI_NIFTI_DIR, f"sub-{pid}_ses-01")
        os.makedirs(out_dir, exist_ok=True)
        output = os.path.join(out_dir, f"sub-{pid}_dwi.nii.gz")
    if output is None:
        parser.error("--output is required (or provide --patient with DWI_NIFTI_DIR in config)")

    dicom_folder_to_nifti_dwi_from_csv_with_skip_and_plot(
        csv_path=csv_path,
        target_series_desc=series_desc,
        out_nii_path=output,
        plot=not args.no_plot,
        bval_prefer=args.bval_prefer,
        bval_min=args.bval_min,
    )


if __name__ == "__main__":
    main()
