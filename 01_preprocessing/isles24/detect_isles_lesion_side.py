#!/usr/bin/env python3
"""detect_isles_lesion_side.py

For every ISLES24 patient, classify the lesion side from the 5-class label map:
  - "Left"  if >= 90% of lesion voxels are on the patient's LEFT hemisphere
  - "Right" if >= 90% of lesion voxels are on the patient's RIGHT hemisphere
  - ""      (empty) if lesions are bilateral (both sides have >= 10% of voxels
            AND >= 500 absolute voxels)

Output: JSON matching the SUS lesion.json format —
  { "sub-stroke0001": "Right", "sub-stroke0002": "Left", "sub-stroke0003": "", ... }

The detection uses the direction matrix of the NIfTI to figure out which voxel-
array side is patient's left vs right. ISLES24 NCCT is RAS-oriented so high X
voxel index = patient's RIGHT, but the script generalises in case any future
case is flipped.
"""

import os
import re
import glob
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


LABEL_DIR = os.path.join(os.environ.get("ISLES24_WORK_ROOT", "/path/to/isles24-work"), "5_Class_Labels")
OUT_JSON  = os.path.join(os.environ.get("PREPROC_REPO_ROOT", "/path/to/preprocessing"), "scripts", "CLB_filtering", "lesion_isles.json")

# Classification thresholds
UNILATERAL_FRACTION = 0.90  # >= 90% on one side  -> unilateral
BILATERAL_MIN_FRAC  = 0.10  # both sides >= 10%   -> bilateral
BILATERAL_MIN_VOX   = 500   #   AND both sides >= 500 voxels


def classify_one(label_path):
    img = sitk.ReadImage(str(label_path))
    arr = sitk.GetArrayFromImage(img)                       # (Z, Y, X)

    # Determine whether voxel-X = patient's right or left
    # direction is 9-tuple (row-major). The first row is the voxel-X column.
    # We need the first component (anatomical X) of the voxel-X direction.
    d = img.GetDirection()
    # SimpleITK direction layout: d = (d00, d01, d02, d10, d11, d12, d20, d21, d22)
    # Voxel axis i mapped to anatomical = (d[i], d[3+i], d[6+i])  (column i)
    # For voxel-X axis (column 0): anatomical_x_component = d[0]
    anat_x_per_voxel_x = d[0]
    voxel_x_high_is_patient_right = (anat_x_per_voxel_x > 0)

    fg_mask = (arr >= 1) & (arr <= 6)
    n_total = int(fg_mask.sum())
    if n_total == 0:
        return {"side": "", "reason": "no foreground", "n_total": 0,
                "n_low": 0, "n_high": 0, "frac_low": 0.0, "frac_high": 0.0,
                "ras": voxel_x_high_is_patient_right}

    xs = np.where(fg_mask)[2]
    midline = arr.shape[2] / 2.0
    n_low  = int((xs <  midline).sum())
    n_high = int((xs >= midline).sum())
    frac_low  = n_low / n_total
    frac_high = n_high / n_total

    # Map voxel-array sides to patient anatomical sides
    if voxel_x_high_is_patient_right:
        n_patient_right, n_patient_left = n_high, n_low
        frac_patient_right, frac_patient_left = frac_high, frac_low
    else:
        n_patient_right, n_patient_left = n_low, n_high
        frac_patient_right, frac_patient_left = frac_low, frac_high

    # Decide
    bilateral = (
        (frac_patient_left  >= BILATERAL_MIN_FRAC and n_patient_left  >= BILATERAL_MIN_VOX) and
        (frac_patient_right >= BILATERAL_MIN_FRAC and n_patient_right >= BILATERAL_MIN_VOX)
    )

    if bilateral:
        side, reason = "", "bilateral"
    elif frac_patient_right >= UNILATERAL_FRACTION:
        side, reason = "Right", "right-dominant"
    elif frac_patient_left  >= UNILATERAL_FRACTION:
        side, reason = "Left", "left-dominant"
    else:
        # In between thresholds (e.g. 85% vs 15%) — still call the majority side
        side = "Right" if frac_patient_right > frac_patient_left else "Left"
        reason = "majority"

    return {
        "side": side,
        "reason": reason,
        "n_total": n_total,
        "n_patient_left":  n_patient_left,
        "n_patient_right": n_patient_right,
        "frac_patient_left":  round(frac_patient_left,  4),
        "frac_patient_right": round(frac_patient_right, 4),
        "ras": voxel_x_high_is_patient_right,
    }


def main():
    files = sorted(glob.glob(os.path.join(LABEL_DIR, "sub-stroke*_5class_map.nii.gz")))
    print(f"Found {len(files)} ISLES 5-class label files.\n")

    mapping = {}
    audit = []
    for f in files:
        pid = re.fullmatch(r"(sub-stroke\d+)_5class_map\.nii\.gz",
                           os.path.basename(f)).group(1)
        res = classify_one(f)
        mapping[pid] = res["side"]
        audit.append({"pid": pid, **res})

        side_str = res["side"] if res["side"] else "(bilateral/empty)"
        print(f"  {pid}  {side_str:>8s}  reason={res['reason']:<20s}  "
              f"L={res['n_patient_left']:>6d}({res['frac_patient_left']*100:5.1f}%)  "
              f"R={res['n_patient_right']:>6d}({res['frac_patient_right']*100:5.1f}%)  "
              f"total={res['n_total']}")

    # Sorted by patient id, formatted exactly like SUS lesion.json
    with open(OUT_JSON, "w") as fp:
        json.dump({k: mapping[k] for k in sorted(mapping)}, fp, indent=4)
    print(f"\nWrote -> {OUT_JSON}")

    # Side summary
    n_left  = sum(1 for v in mapping.values() if v == "Left")
    n_right = sum(1 for v in mapping.values() if v == "Right")
    n_empty = sum(1 for v in mapping.values() if v == "")
    print(f"\nSummary: Left={n_left}  Right={n_right}  bilateral/empty={n_empty}")

    # Save audit alongside for the subagent to cross-check
    audit_path = OUT_JSON.replace(".json", "_audit.json")
    with open(audit_path, "w") as fp:
        json.dump(audit, fp, indent=2)
    print(f"Audit  -> {audit_path}")


if __name__ == "__main__":
    main()
