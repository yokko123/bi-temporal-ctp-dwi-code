"""
s00_extract_dicom_headers.py
============================
Extract DICOM header metadata from patient folder trees and write to CSV.

Walks directory trees, reads one representative DICOM per folder (or every
file), and collects user-specified tags into a flat CSV table.  Useful as the
first step in the preprocessing pipeline so that downstream scripts can look
up series folders by SeriesDescription, Modality, etc.

Standalone usage
----------------
    python s00_extract_dicom_headers.py \
        --roots /data/patient_001/DICOM /data/patient_002/DICOM \
        --output-csv dicom_headers.csv \
        --one-row-per folder

Pipeline usage
--------------
    from registration.s00_extract_dicom_headers import extract_dicom_headers
"""

import os
import argparse

import pydicom
import pandas as pd
from pydicom.misc import is_dicom

try:
    from config import DICOM_ROOTS, DICOM_HEADERS_CSV
except ImportError:
    DICOM_ROOTS = None
    DICOM_HEADERS_CSV = "dicom_headers.csv"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def safe_get(ds, name, default=""):
    """Get DICOM attribute as string safely."""
    val = getattr(ds, name, default)
    if val is None:
        return ""
    try:
        return str(val)
    except Exception:
        return ""


def find_one_dicom_file(folder_path):
    """Return one dicom file path inside folder_path, or None."""
    try:
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            if os.path.isfile(fpath):
                try:
                    if is_dicom(fpath):
                        return fpath
                except Exception:
                    continue
    except Exception:
        return None
    return None


def folder_stats(folder_path):
    """Count dicom files + total bytes in a folder."""
    num_dicom = 0
    total_bytes = 0
    try:
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                total_bytes += os.path.getsize(fpath)
            except Exception:
                pass
            try:
                if is_dicom(fpath):
                    num_dicom += 1
            except Exception:
                pass
    except Exception:
        pass
    return num_dicom, total_bytes


# ------------------------------------------------------------------
# Main extraction function
# ------------------------------------------------------------------

def extract_dicom_headers(
    roots,
    output_csv="dicom_headers.csv",
    tags=None,
    one_row_per="folder",  # "folder" (recommended) or "file"
    force=True
):
    """
    roots: str or list[str]  (patient root folder or multiple folders)
    tags: list of DICOM attribute names e.g. ["Modality","SeriesDescription","SeriesInstanceUID",...]
          If None -> uses a sensible default.
    one_row_per:
        - "folder": one representative DICOM per folder that contains DICOM files
        - "file":   one row per dicom file (can be huge)
    """
    if isinstance(roots, str):
        roots = [roots]

    if tags is None:
        tags = [
            "PatientID", "StudyDate", "StudyTime",
            "Modality", "Manufacturer",
            "SeriesDescription", "SeriesNumber",
            "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID",
            "ImageType",
            "Rows", "Columns",
            "PixelSpacing", "SliceThickness", "SpacingBetweenSlices",
            "GantryDetectorTilt",
            "TemporalPositionIdentifier", "NumberOfTemporalPositions",
            "TriggerTime", "AcquisitionTime",
            "ImagePositionPatient", "ImageOrientationPatient",
            "ConvolutionKernel", "KVP"
        ]

    rows = []

    for root_path in roots:
        if not os.path.isdir(root_path):
            print(f"[WARN] Not a directory: {root_path}")
            continue

        # Walk all nested folders
        for cur_root, dirs, files in os.walk(root_path):
            if one_row_per == "folder":
                # only process folders which contain at least one dicom file
                rep = find_one_dicom_file(cur_root)
                if rep is None:
                    continue

                num_dicom, total_bytes = folder_stats(cur_root)

                try:
                    ds = pydicom.dcmread(rep, stop_before_pixels=True, force=force)
                except Exception as e:
                    print(f"[WARN] Failed reading {rep}: {e}")
                    continue

                row = {
                    "Folder": cur_root,
                    "NumDicomFiles": num_dicom,
                }
                for t in tags:
                    row[t] = safe_get(ds, t)

                rows.append(row)

            elif one_row_per == "file":
                for fname in files:
                    fp = os.path.join(cur_root, fname)
                    if not os.path.isfile(fp):
                        continue
                    try:
                        if not is_dicom(fp):
                            continue
                    except Exception:
                        continue

                    try:
                        ds = pydicom.dcmread(fp, stop_before_pixels=True, force=force)
                    except Exception as e:
                        print(f"[WARN] Failed reading {fp}: {e}")
                        continue

                    row = {"InputRoot": root_path, "Folder": cur_root, "FilePath": fp}
                    for t in tags:
                        row[t] = safe_get(ds, t)
                    rows.append(row)
            else:
                raise ValueError("one_row_per must be 'folder' or 'file'")

    if not rows:
        raise RuntimeError("No DICOM files found under provided roots.")

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} rows to {output_csv}")
    return df


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main(cli_args=None):
    parser = argparse.ArgumentParser(
        description="Extract DICOM header tags from folder trees into a CSV."
    )
    parser.add_argument(
        "--roots", nargs="+", required=True,
        help="One or more root directories containing DICOM files."
    )
    parser.add_argument(
        "--output-csv", default=DICOM_HEADERS_CSV,
        help="Path for the output CSV file (default: %(default)s)."
    )
    parser.add_argument(
        "--one-row-per", choices=["folder", "file"], default="folder",
        help="Generate one row per folder (default) or per file."
    )
    args = parser.parse_args(cli_args)

    extract_dicom_headers(
        roots=args.roots,
        output_csv=args.output_csv,
        one_row_per=args.one_row_per,
    )


if __name__ == "__main__":
    main()
