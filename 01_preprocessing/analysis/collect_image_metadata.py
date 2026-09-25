"""
collect_image_metadata.py
=========================
Collect size, spacing, and origin for all modalities (NCCT, CTP, CTP label,
DWI, DWI mask, ADC) across all patients and save to CSV.

Standalone usage
----------------
    python collect_image_metadata.py
    python collect_image_metadata.py --output metadata.csv

Pipeline usage
--------------
    from analysis.collect_image_metadata import get_info
"""

import argparse
import csv
import os
from pathlib import Path

import SimpleITK as sitk

try:
    from config import REG_ROOT
except ImportError:
    REG_ROOT = os.environ.get("SUS_WORK_ROOT", "/path/to/sus-work")

CURATED = Path(os.environ.get("SUS_CURATED_ROOT", "/path/to/Curated-SUS-2025"))
DERIV = CURATED / "derivatives"
RAW = CURATED / "raw_data_nifti"

# (label, session, path_template relative to base_dir, base_dir_key)
MODALITIES = [
    ("NCCT",      "ses-01", "raw",  "{raw}/sub-stroke_{pid}/ses-01/sub-stroke_{pid}_ses-01_ncct_skull_stripped.nii.gz"),
    ("CTP_4D",    "ses-01", "deriv", "{deriv}/sub-stroke_{pid}/ses-01/sub-stroke_{pid}_ses-01_space-ncct_ctp.nii.gz"),
    ("CTP_Label", "ses-01", "deriv", "{deriv}/sub-stroke_{pid}/ses-01/sub-stroke_{pid}_ses-01_space-ncct_ctp_lesion_msk_3class.nii.gz"),
    ("DWI",       "ses-02", "deriv", "{deriv}/sub-stroke_{pid}/ses-02/sub-stroke_{pid}_ses-02_space-ncct_dwi.nii.gz"),
    ("DWI_Mask",  "ses-02", "deriv", "{deriv}/sub-stroke_{pid}/ses-02/sub-stroke_{pid}_ses-02_space-ncct_dwi_lesion_msk.nii.gz"),
    ("ADC",       "ses-02", "deriv", "{deriv}/sub-stroke_{pid}/ses-02/sub-stroke_{pid}_ses-02_space-ncct_adc.nii.gz"),
]


def get_info(filepath):
    """Read image and return size, spacing, origin as strings."""
    img = sitk.ReadImage(str(filepath))
    size = img.GetSize()
    spacing = tuple(round(s, 4) for s in img.GetSpacing())
    origin = tuple(round(o, 4) for o in img.GetOrigin())
    return size, spacing, origin


def main():
    parser = argparse.ArgumentParser(description="Collect image metadata to CSV")
    parser.add_argument("--output", type=Path,
                        default=Path(REG_ROOT) / "registered_patient_metadata.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    # Discover patients from derivatives folder
    patient_dirs = sorted(DERIV.glob("sub-stroke_*"))
    pids = [d.name.replace("sub-stroke_", "") for d in patient_dirs]
    print(f"Found {len(pids)} patients")

    rows = []
    for i, pid in enumerate(pids):
        print(f"[{i+1}/{len(pids)}] sub-stroke_{pid}")
        for mod_name, session, base_key, path_tmpl in MODALITIES:
            fpath = Path(path_tmpl.format(pid=pid, deriv=DERIV, raw=RAW))
            if not fpath.exists():
                rows.append({
                    "patient_id": pid,
                    "modality": mod_name,
                    "session": session,
                    "exists": False,
                    "size_x": "", "size_y": "", "size_z": "", "size_t": "",
                    "spacing_x": "", "spacing_y": "", "spacing_z": "", "spacing_t": "",
                    "origin_x": "", "origin_y": "", "origin_z": "", "origin_t": "",
                })
                continue

            try:
                size, spacing, origin = get_info(fpath)
            except Exception as e:
                print(f"  [ERR] {mod_name}: {e}")
                rows.append({
                    "patient_id": pid, "modality": mod_name, "session": session,
                    "exists": True,
                    "size_x": "ERR", "size_y": "", "size_z": "", "size_t": "",
                    "spacing_x": "", "spacing_y": "", "spacing_z": "", "spacing_t": "",
                    "origin_x": "", "origin_y": "", "origin_z": "", "origin_t": "",
                })
                continue

            ndim = len(size)
            row = {
                "patient_id": pid,
                "modality": mod_name,
                "session": session,
                "exists": True,
                "size_x": size[0], "size_y": size[1], "size_z": size[2],
                "size_t": size[3] if ndim == 4 else "",
                "spacing_x": spacing[0], "spacing_y": spacing[1], "spacing_z": spacing[2],
                "spacing_t": spacing[3] if ndim == 4 else "",
                "origin_x": origin[0], "origin_y": origin[1], "origin_z": origin[2],
                "origin_t": origin[3] if ndim == 4 else "",
            }
            rows.append(row)

    # Write CSV
    fieldnames = [
        "patient_id", "modality", "session", "exists",
        "size_x", "size_y", "size_z", "size_t",
        "spacing_x", "spacing_y", "spacing_z", "spacing_t",
        "origin_x", "origin_y", "origin_z", "origin_t",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
