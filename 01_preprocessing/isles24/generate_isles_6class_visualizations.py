#!/usr/bin/env python3
"""generate_isles_6class_visualizations.py

ISLES24 6-class outcome-map visualizations.

Mirrors generate_6class_visualizations.py (SUS) but adapted to ISLES:
  - 6 classes. Numeric labels match SUS:
    {1=core_fi, 2=core_brain, 3=pen_fi, 4=pen_brain, 5=CLB_brain, 6=nhb_fi}
  - Reads from ISLES24/analysis/, which is the union folder containing
    6-class maps for every patient. The 22 cases that lack a CLB reference
    have been copied in with the same _6class_map suffix but simply contain
    no voxels with label 5 — they render identically to the old 5-class view.
  - Backgrounds:
      * DWI  (always available — under ISLES source derivatives)
      * CTP  (4D, from isles24_curated; only if assembled)

For each patient, saves one multi-slice PNG per available background under
   <output_dir>/<pid>/<pid>_6class_<modality>.png
"""

import os
import re
import glob
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from tqdm import tqdm


# --------------------------------------------------
# Defaults
# --------------------------------------------------
ISLES_BASE       = os.environ.get("ISLES24_WORK_ROOT", "/path/to/isles24-work")
ISLES_SOURCE     = os.environ.get(
    "ISLES24_RELEASE_DIR", "/path/to/ISLES24/version_7/train/derivatives")
CTP_4D_BASE      = os.path.join(ISLES_BASE, "isles24_curated", "derivatives")
LABEL_DIR        = os.path.join(ISLES_BASE, "analysis")
DEFAULT_OUT_DIR  = (os.path.join(os.environ.get("ISLES24_WORK_ROOT", "/path/to/isles24-work"), "isles24_6class_visualizations"))
DEFAULT_TIMEPT   = 20  # CTP timepoint to use as background

# --------------------------------------------------
# Colors / legend (matches SUS 6-class viz)
# --------------------------------------------------
# colormap: index 0 transparent, then per-label color (1..6)
CMAP_COLORS = ["none", "red", "blue", "orange", "lightgreen", "green", "yellow"]
OUTCOME_CMAP = ListedColormap(CMAP_COLORS)

LEGEND_ELEMENTS = [
    Line2D([0], [0], color="red",        lw=4, label="Core → FI"),
    Line2D([0], [0], color="blue",       lw=4, label="Core → Brain"),
    Line2D([0], [0], color="orange",     lw=4, label="Penumbra → FI"),
    Line2D([0], [0], color="lightgreen", lw=4, label="Penumbra → Brain"),
    Line2D([0], [0], color="green",      lw=4, label="CLB → Brain"),
    Line2D([0], [0], color="yellow",     lw=4, label="NHB → FI"),
]


# --------------------------------------------------
# I/O
# --------------------------------------------------
def load_nifti(path):
    """Load NIfTI; transpose to (Z, Y, X) or (T, Z, Y, X)."""
    img  = nib.load(str(path))
    data = img.get_fdata().astype(np.float32)
    if data.ndim == 3:
        return data.transpose(2, 1, 0)
    elif data.ndim == 4:
        return data.transpose(3, 2, 1, 0)
    return data


def discover_patients(label_dir):
    """Find every patient who has a 6-class label file."""
    pids = []
    for f in sorted(glob.glob(os.path.join(label_dir, "*_6class_map.nii.gz"))):
        m = re.fullmatch(r"(sub-stroke\d+)_6class_map\.nii\.gz",
                         os.path.basename(f))
        if m:
            pids.append(m.group(1))
    return pids


def patient_paths(pid):
    """Return dict of available file paths for one patient."""
    paths = {
        "label": os.path.join(LABEL_DIR, f"{pid}_6class_map.nii.gz"),
        "dwi":   os.path.join(ISLES_SOURCE, pid, "ses-02",
                              f"{pid}_ses-02_space-ncct_dwi.nii.gz"),
        "ctp":   os.path.join(CTP_4D_BASE, pid, "ses-01",
                              f"{pid}_ses-01_space-ncct_ctp.nii.gz"),
    }
    return paths


# --------------------------------------------------
# Plotting
# --------------------------------------------------
def plot_6class_visualization(pid, t_idx, output_dir):
    """Render one patient's overlays for available backgrounds."""
    paths = patient_paths(pid)
    if not os.path.exists(paths["label"]):
        return False, "missing label"

    outcome_arr = load_nifti(paths["label"])

    # Configs: (key, title, vmin, vmax, is_4d)
    configs = []
    if os.path.exists(paths["dwi"]):
        configs.append(("dwi", "DWI", 0, 1000, False))
    if os.path.exists(paths["ctp"]):
        configs.append(("ctp", f"CTP (t={t_idx})", 0, 255, True))

    if not configs:
        return False, "no backgrounds available"

    out_pat_dir = os.path.join(output_dir, pid)
    os.makedirs(out_pat_dir, exist_ok=True)

    for key, title, vmin, vmax, is_4d in configs:
        bg = load_nifti(paths[key])
        n_slices = bg.shape[1] if is_4d else bg.shape[0]

        # Align label to background slice count if there's a mismatch in Z
        n_z = min(n_slices, outcome_arr.shape[0])

        cols = 10
        rows = int(np.ceil(n_z / cols))
        fig, axes = plt.subplots(rows, cols,
                                 figsize=(20, 2 * rows + 1),
                                 facecolor="black")
        fig.suptitle(f"{pid} | {title}",
                     color="white", fontsize=18, fontweight="bold")
        axes = axes.flatten() if rows > 1 else axes

        last_used = -1
        for i in range(n_z):
            ax = axes[i]
            bg_slice    = bg[t_idx, i] if is_4d else bg[i]
            label_slice = outcome_arr[i]

            # Flip both vertically so the displayed orientation matches the
            # convention used elsewhere in the pipeline.
            bg_slice    = np.flipud(bg_slice)
            label_slice = np.flipud(label_slice)

            ax.imshow(bg_slice, cmap="gray", vmin=vmin, vmax=vmax)

            if np.any(label_slice > 0):
                masked = np.ma.masked_where(label_slice == 0, label_slice)
                ax.imshow(masked, cmap=OUTCOME_CMAP, alpha=0.6,
                          vmin=0, vmax=6)

            ax.set_title(f"z={i}", fontsize=8, color="white")
            ax.axis("off")
            last_used = i

        for j in range(last_used + 1, len(axes)):
            axes[j].axis("off")

        fig.legend(handles=LEGEND_ELEMENTS, loc="lower center", ncol=6,
                   bbox_to_anchor=(0.5, 0.01), frameon=False,
                   labelcolor="white", fontsize=12)
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])

        out_path = os.path.join(out_pat_dir, f"{pid}_6class_{key}.png")
        fig.savefig(out_path, dpi=150, facecolor="black",
                    bbox_inches="tight")
        plt.close(fig)

    return True, "ok"


# --------------------------------------------------
# CLI
# --------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Generate 6-class outcome visualizations for ISLES24 patients.")
    ap.add_argument("--output-dir", default=DEFAULT_OUT_DIR,
                    help="Where to write per-patient subfolders.")
    ap.add_argument("--timepoint", type=int, default=DEFAULT_TIMEPT,
                    help="CTP timepoint index used as background.")
    ap.add_argument("--patients", default=None,
                    help="Comma-separated subset of patient IDs (e.g. sub-stroke0001,sub-stroke0002).")
    ap.add_argument("--start-idx", type=int, default=None,
                    help="1-based start index.")
    ap.add_argument("--end-idx",   type=int, default=None,
                    help="1-based end index (inclusive).")
    ap.add_argument("--dry-run", action="store_true",
                    help="List target patients and exit.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("ISLES24 6-class outcome visualization")
    print("=" * 70)

    all_pids = discover_patients(LABEL_DIR)
    print(f"\nFound {len(all_pids)} patients with 6-class label files.")
    if not all_pids:
        print("Nothing to do.")
        return

    if args.patients:
        wanted = [p.strip() for p in args.patients.split(",") if p.strip()]
        selected = [p for p in wanted if p in all_pids]
        missing  = sorted(set(wanted) - set(selected))
        if missing:
            print(f"Warning: not in label directory: {missing}")
    else:
        start = (args.start_idx - 1) if args.start_idx else 0
        end   = args.end_idx if args.end_idx else len(all_pids)
        selected = all_pids[start:end]

    print(f"Processing {len(selected)} patients.")

    if args.dry_run:
        for i, pid in enumerate(selected, 1):
            p = patient_paths(pid)
            avail = []
            if os.path.exists(p["dwi"]): avail.append("DWI")
            if os.path.exists(p["ctp"]): avail.append("CTP")
            print(f"  {i:3d}. {pid}  backgrounds={avail or ['none']}")
        return

    n_ok = n_skip = n_fail = 0
    for pid in tqdm(selected, desc="patients"):
        try:
            ok, reason = plot_6class_visualization(pid, args.timepoint, args.output_dir)
            if ok:
                n_ok += 1
            else:
                n_skip += 1
                print(f"  ⚠ skip {pid}: {reason}")
        except Exception as e:
            n_fail += 1
            print(f"  ✗ {pid}: {e}")
            import traceback; traceback.print_exc()

    print(f"\n{'=' * 70}")
    print(f"ok={n_ok}  skipped={n_skip}  failed={n_fail}  total={len(selected)}")
    print(f"Output -> {args.output_dir}")


if __name__ == "__main__":
    main()
