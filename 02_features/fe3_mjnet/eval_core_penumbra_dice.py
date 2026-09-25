"""
Compute core/penumbra Dice for mJ-Net predictions (saved as NIfTI volumes)
against the curated 3-class lesion masks.

Class map (matches predict.py):
    0 = background
    1 = brain
    2 = penumbra
    3 = core

Post-processing options (per class, after argmax):
    --median-radius   : 2D median filter radius applied per slice (smoothing)
    --closing-radius  : 3D binary closing radius for penumbra+core
    --min-size-core   : remove core connected components smaller than this (voxels)
    --min-size-penum  : remove penumbra connected components smaller than this (voxels)

Usage:
    python eval_core_penumbra_dice.py \
        --pred_dir checkpoints/cv_5fold_20260313_125726/test_predictions \
        --derivatives_dir $SUS_CURATED_ROOT \
        --output_csv core_penumbra_dice.csv \
        --closing-radius 1 --min-size-core 20 --min-size-penum 50
"""

import os
import re
import argparse
import glob
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects


CLASS_NAMES = {0: "background", 1: "brain", 2: "penumbra", 3: "core"}


def dice(p_mask: np.ndarray, g_mask: np.ndarray) -> float:
    p = p_mask.astype(bool)
    g = g_mask.astype(bool)
    s = p.sum() + g.sum()
    if s == 0:
        return float("nan")  # neither pred nor gt has this class -> undefined
    return float(2.0 * np.logical_and(p, g).sum() / s)


def postprocess(
    seg: np.ndarray,
    median_radius: int = 0,
    closing_radius: int = 0,
    min_size_core: int = 0,
    min_size_penum: int = 0,
    core_dilate_iters: int = 0,
    core_inside_penumbra_only: bool = False,
) -> np.ndarray:
    """Apply optional smoothing to an integer label volume."""
    out = seg.copy()

    if median_radius > 0:
        size = 2 * median_radius + 1
        for z in range(out.shape[0]):
            out[z] = ndi.median_filter(out[z], size=size)

    if closing_radius > 0:
        struct = ndi.generate_binary_structure(3, 1)
        struct = ndi.iterate_structure(struct, closing_radius)
        for c in (2, 3):
            m = out == c
            m_closed = ndi.binary_closing(m, structure=struct)
            new_voxels = m_closed & ~m
            paintable = (out == 0) | (out == 1)
            out[new_voxels & paintable] = c

    if core_dilate_iters > 0:
        # Paper §5: "annotations might leave out small regions of core spread in
        # the penumbra". Dilate predicted core into adjacent predicted penumbra.
        struct = ndi.generate_binary_structure(3, 1)
        core = out == 3
        core_dil = ndi.binary_dilation(core, structure=struct, iterations=core_dilate_iters)
        promote = core_dil & (out == 2)
        out[promote] = 3

    if core_inside_penumbra_only:
        # Demote core voxels that have no penumbra in their 1-voxel neighborhood
        # (kills floating core FPs that sit in plain brain).
        struct = ndi.generate_binary_structure(3, 1)
        pen = out == 2
        pen_near = ndi.binary_dilation(pen, structure=struct, iterations=1)
        out[(out == 3) & ~pen_near] = 1

    if min_size_core > 0:
        core = out == 3
        core_clean = remove_small_objects(core, min_size=min_size_core, connectivity=1)
        removed = core & ~core_clean
        out[removed] = 1

    if min_size_penum > 0:
        pen = out == 2
        pen_clean = remove_small_objects(pen, min_size=min_size_penum, connectivity=1)
        removed = pen & ~pen_clean
        out[removed] = 1

    return out


def find_pairs(pred_dir: str, derivatives_dir: str, exclude: list = None):
    exclude = set(exclude or [])
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "sub-stroke_*_prediction.nii.gz")))
    pairs = []
    for pf in pred_files:
        m = re.match(r"(sub-stroke_\d{2}_\d{3})_prediction\.nii\.gz", os.path.basename(pf))
        if not m:
            continue
        subj = m.group(1)
        if subj in exclude:
            print(f"[skip] {subj} excluded by user")
            continue
        gt = os.path.join(
            derivatives_dir, "derivatives", subj, "ses-01",
            f"{subj}_ses-01_space-ncct_ctp_lesion_msk_3class.nii.gz",
        )
        if not os.path.isfile(gt):
            print(f"[skip] no GT for {subj}: {gt}")
            continue
        # If probability map exists alongside the prediction, capture it.
        prob_path = os.path.join(os.path.dirname(pf), f"{subj}_probs.nii.gz")
        if not os.path.isfile(prob_path):
            prob_path = None
        pairs.append((subj, pf, gt, prob_path))
    return pairs


def argmax_from_probs(
    probs: np.ndarray,
    T_core: float = None,
    T_pen: float = None,
    w_core: float = 1.0,
    w_pen: float = 1.0,
) -> np.ndarray:
    """Decide labels from softmax probs with optional class thresholds / weights.

    probs : (C, Z, Y, X) softmax map, C=4 in order [bg, brain, pen, core].

    Decision rules (in order, first wins):
      1. if T_core is not None and probs[3] > T_core -> core
      2. if T_pen  is not None and probs[2] > T_pen  -> penumbra
      3. else argmax( probs * [1,1,w_pen,w_core] )

    Either-or: pass T_core / T_pen for hard threshold mode, or pass w_core/w_pen
    for prior-shift mode.
    """
    p_bg, p_brain, p_pen, p_core = probs[0], probs[1], probs[2], probs[3]

    if T_core is not None or T_pen is not None:
        # Hard threshold mode (closest analog of the paper's grayscale thresholds).
        out = np.argmax(probs, axis=0).astype(np.int16)
        if T_pen is not None:
            out[p_pen > T_pen] = 2
        if T_core is not None:
            out[p_core > T_core] = 3  # core overrides penumbra
        return out

    # Prior-shift mode.
    weighted = probs.copy()
    weighted[2] *= w_pen
    weighted[3] *= w_core
    return np.argmax(weighted, axis=0).astype(np.int16)


def evaluate_one(
    pred_path: str,
    gt_path: str,
    pp_kwargs: dict,
    prob_path: str = None,
    T_core: float = None,
    T_pen: float = None,
    w_core: float = 1.0,
    w_pen: float = 1.0,
):
    gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_path)).astype(np.int16)

    # If a probability map is available AND threshold/weights requested, use it.
    using_probs = prob_path is not None and (
        T_core is not None or T_pen is not None or w_core != 1.0 or w_pen != 1.0
    )
    if using_probs:
        probs = sitk.GetArrayFromImage(sitk.ReadImage(prob_path)).astype(np.float32)
        # SimpleITK stores 4D as (Z, Y, X, C) component-last when written from (C, Z, Y, X).
        # Try to normalise to (C, Z, Y, X).
        if probs.ndim == 4 and probs.shape[-1] == 4 and probs.shape[0] != 4:
            probs = np.transpose(probs, (3, 0, 1, 2))
        pred = argmax_from_probs(probs, T_core=T_core, T_pen=T_pen,
                                 w_core=w_core, w_pen=w_pen)
    else:
        pred = sitk.GetArrayFromImage(sitk.ReadImage(pred_path)).astype(np.int16)

    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")

    raw_metrics = {
        "dice_penumbra_raw": dice(pred == 2, gt == 2),
        "dice_core_raw":     dice(pred == 3, gt == 3),
        "dice_lesion_raw":   dice((pred == 2) | (pred == 3), (gt == 2) | (gt == 3)),
    }

    pp = postprocess(pred, **pp_kwargs)
    pp_metrics = {
        "dice_penumbra_pp": dice(pp == 2, gt == 2),
        "dice_core_pp":     dice(pp == 3, gt == 3),
        "dice_lesion_pp":   dice((pp == 2) | (pp == 3), (gt == 2) | (gt == 3)),
    }

    voxel_counts = {
        "gt_penumbra_vox":   int((gt == 2).sum()),
        "gt_core_vox":       int((gt == 3).sum()),
        "pred_penumbra_vox": int((pred == 2).sum()),
        "pred_core_vox":     int((pred == 3).sum()),
        "pp_penumbra_vox":   int((pp == 2).sum()),
        "pp_core_vox":       int((pp == 3).sum()),
    }

    return {**raw_metrics, **pp_metrics, **voxel_counts}, pp


def nanmean(xs):
    xs = np.asarray(xs, dtype=np.float64)
    return float(np.nanmean(xs)) if np.isfinite(xs).any() else float("nan")


def nanmedian(xs):
    xs = np.asarray(xs, dtype=np.float64)
    return float(np.nanmedian(xs)) if np.isfinite(xs).any() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--derivatives_dir", required=True,
                    help="Folder containing the 'derivatives/' tree.")
    ap.add_argument("--output_csv", default="core_penumbra_dice.csv")
    ap.add_argument("--save_pp_dir", default=None,
                    help="If set, write post-processed prediction NIfTIs here.")
    ap.add_argument("--median-radius", type=int, default=0)
    ap.add_argument("--closing-radius", type=int, default=0)
    ap.add_argument("--min-size-core", type=int, default=0)
    ap.add_argument("--min-size-penum", type=int, default=0)
    ap.add_argument("--core-dilate-iters", type=int, default=0,
                    help="Dilate core into adjacent penumbra (paper-inspired).")
    ap.add_argument("--core-inside-penumbra-only", action="store_true",
                    help="Drop core voxels not adjacent to any predicted penumbra.")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="Patient IDs to exclude.")
    ap.add_argument("--T-core", type=float, default=None,
                    help="If set, label voxels with p_core > T_core as core (needs *_probs.nii.gz).")
    ap.add_argument("--T-pen", type=float, default=None,
                    help="If set, label voxels with p_penumbra > T_pen as penumbra.")
    ap.add_argument("--w-core", type=float, default=1.0,
                    help="Multiplicative prior shift on p_core before argmax.")
    ap.add_argument("--w-pen", type=float, default=1.0,
                    help="Multiplicative prior shift on p_penumbra before argmax.")
    args = ap.parse_args()

    pp_kwargs = dict(
        median_radius=args.median_radius,
        closing_radius=args.closing_radius,
        min_size_core=args.min_size_core,
        min_size_penum=args.min_size_penum,
        core_dilate_iters=args.core_dilate_iters,
        core_inside_penumbra_only=args.core_inside_penumbra_only,
    )

    pairs = find_pairs(args.pred_dir, args.derivatives_dir, exclude=args.exclude)
    print(f"Found {len(pairs)} (pred, gt) pairs")
    print(f"Post-processing: {pp_kwargs}")
    print(f"Prob-mode: T_core={args.T_core} T_pen={args.T_pen} "
          f"w_core={args.w_core} w_pen={args.w_pen}")

    if args.save_pp_dir:
        os.makedirs(args.save_pp_dir, exist_ok=True)

    rows = []
    for subj, pf, gf, prob_path in pairs:
        try:
            metrics, pp_vol = evaluate_one(
                pf, gf, pp_kwargs,
                prob_path=prob_path,
                T_core=args.T_core, T_pen=args.T_pen,
                w_core=args.w_core, w_pen=args.w_pen,
            )
        except Exception as e:
            print(f"[err] {subj}: {e}")
            continue

        if args.save_pp_dir:
            img = sitk.ReadImage(pf)
            out_img = sitk.GetImageFromArray(pp_vol.astype(np.int16))
            out_img.CopyInformation(img)
            sitk.WriteImage(out_img, os.path.join(args.save_pp_dir, f"{subj}_prediction_pp.nii.gz"))

        rows.append({"subject": subj, **metrics})
        print(
            f"{subj}: "
            f"core raw={metrics['dice_core_raw']:.4f} pp={metrics['dice_core_pp']:.4f} | "
            f"pen raw={metrics['dice_penumbra_raw']:.4f} pp={metrics['dice_penumbra_pp']:.4f} | "
            f"les raw={metrics['dice_lesion_raw']:.4f} pp={metrics['dice_lesion_pp']:.4f} "
            f"(gt core={metrics['gt_core_vox']}, pen={metrics['gt_penumbra_vox']})"
        )

    if not rows:
        print("No rows.")
        return

    # Write CSV
    import csv
    keys = list(rows[0].keys())
    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.output_csv}")

    # Aggregates
    def col(name):
        return [r[name] for r in rows]

    print("\n=== Aggregate Dice (mean ± std, median) ===")
    for name in ["dice_core_raw", "dice_core_pp",
                 "dice_penumbra_raw", "dice_penumbra_pp",
                 "dice_lesion_raw", "dice_lesion_pp"]:
        v = np.array(col(name), dtype=np.float64)
        mean = nanmean(v)
        med = nanmedian(v)
        std = float(np.nanstd(v))
        n = int(np.isfinite(v).sum())
        print(f"  {name:24s}  n={n:3d}  mean={mean:.4f}  std={std:.4f}  median={med:.4f}")

    # Also report "patients with GT lesion only" — undefined Dice when GT class absent
    print("\n=== Restricted to patients where GT class is present ===")
    pen_pres = [r for r in rows if r["gt_penumbra_vox"] > 0]
    core_pres = [r for r in rows if r["gt_core_vox"] > 0]
    print(f"  penumbra present in GT: {len(pen_pres)}/{len(rows)}")
    if pen_pres:
        v_raw = np.array([r["dice_penumbra_raw"] for r in pen_pres])
        v_pp  = np.array([r["dice_penumbra_pp"]  for r in pen_pres])
        print(f"    penumbra dice raw mean={np.nanmean(v_raw):.4f}  pp mean={np.nanmean(v_pp):.4f}")
    print(f"  core     present in GT: {len(core_pres)}/{len(rows)}")
    if core_pres:
        v_raw = np.array([r["dice_core_raw"] for r in core_pres])
        v_pp  = np.array([r["dice_core_pp"]  for r in core_pres])
        print(f"    core     dice raw mean={np.nanmean(v_raw):.4f}  pp mean={np.nanmean(v_pp):.4f}")


if __name__ == "__main__":
    main()
