#!/usr/bin/env python3
"""
Standalone prediction script for the nnU-Net core/penumbra segmenter.

Fetches the 5-fold weights from the Hub if needed, points nnU-Net at them, and
runs inference, so you do not have to set nnUNet_results or remember the
nnUNetv2_predict flags. Writes a 4-class label map
(0 background, 1 brain, 2 penumbra, 3 core).

Input must be preprocessed exactly as in stage 01 of the code repository:
motion-corrected, resampled to 40 frames, registered to the admission NCCT,
skull-stripped, windowed to [0, 100] HU and rescaled to an 8-bit grey range.
All 40 channels are declared NoNormalization, so the model will not behave
sensibly on raw CTP.

Each case is 40 files, CASE_0000.nii.gz ... CASE_0039.nii.gz, in timepoint
order, all in one input folder. Several cases may share that folder.

Usage:
  python predict.py -i INPUT_FOLDER -o OUTPUT_FOLDER

  # single fold, much faster, lower quality
  python predict.py -i INPUT_FOLDER -o OUTPUT_FOLDER --folds 0

  # apply the cross-validated postprocessing nnU-Net selected
  python predict.py -i INPUT_FOLDER -o OUTPUT_FOLDER --postprocess

  # weights already on disk
  python predict.py -i INPUT_FOLDER -o OUTPUT_FOLDER --weights $nnUNet_results
"""

import argparse
import glob
import logging
import os
import re
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

REPO_ID = "yokko123/ctp-core-penumbra-nnunet"
DATASET = "Dataset1152_SUS_CTP_Reg_3DT"
TRAINER_DIR = "nnUNetTrainer__nnUNetPlans__3d_fullres"
NUM_TIMEPOINTS = 40
CLASS_NAMES = {0: "background", 1: "brain", 2: "penumbra", 3: "core"}


def resolve_weights(weights_dir):
    """Return an nnUNet_results-style root holding the trained model."""
    if weights_dir:
        expected = os.path.join(weights_dir, DATASET, TRAINER_DIR)
        if not os.path.isdir(expected):
            raise FileNotFoundError(
                f"Expected {expected}. --weights must point at an nnUNet_results "
                f"root containing {DATASET}/."
            )
        return weights_dir

    from huggingface_hub import snapshot_download
    logging.info(f"Fetching weights from {REPO_ID} (~1.8 GB on first run)")
    return snapshot_download(
        repo_id=REPO_ID,
        allow_patterns=[f"{DATASET}/**"],
    )


def check_input(folder):
    """Fail early and clearly on the mistake everyone makes: wrong channel count."""
    files = sorted(glob.glob(os.path.join(folder, "*_[0-9][0-9][0-9][0-9].nii.gz")))
    if not files:
        raise FileNotFoundError(
            f"No files matching <case>_0000.nii.gz in {folder}."
        )
    cases = {}
    for f in files:
        m = re.match(r"(.+)_(\d{4})\.nii\.gz$", os.path.basename(f))
        if m:
            cases.setdefault(m.group(1), set()).add(int(m.group(2)))

    bad = {c: sorted(ch) for c, ch in cases.items()
           if ch != set(range(NUM_TIMEPOINTS))}
    if bad:
        for c, ch in bad.items():
            logging.error(
                f"{c}: found {len(ch)} channels, need exactly {NUM_TIMEPOINTS} "
                f"({c}_0000.nii.gz ... {c}_{NUM_TIMEPOINTS - 1:04d}.nii.gz)"
            )
        raise ValueError("Input does not satisfy the 40-channel contract.")

    logging.info(f"{len(cases)} case(s), {NUM_TIMEPOINTS} channels each")
    return sorted(cases)


def run(cmd, env):
    logging.info("$ " + " ".join(cmd))
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def summarise(out_dir, cases):
    try:
        import numpy as np
        import SimpleITK as sitk
    except ImportError:
        return
    for c in cases:
        p = os.path.join(out_dir, f"{c}.nii.gz")
        if not os.path.isfile(p):
            continue
        a = sitk.GetArrayFromImage(sitk.ReadImage(p))
        counts = ", ".join(f"{n} {int((a == v).sum())}"
                           for v, n in CLASS_NAMES.items() if v != 0)
        logging.info(f"  {c}: {counts} voxels")


def main():
    ap = argparse.ArgumentParser(
        description="nnU-Net core/penumbra prediction (5-fold ensemble).")
    ap.add_argument("-i", "--input", required=True,
                    help="Folder of <case>_0000.nii.gz ... <case>_0039.nii.gz.")
    ap.add_argument("-o", "--output", required=True, help="Output folder.")
    ap.add_argument("--weights", default=None,
                    help="nnUNet_results root (default: download from the Hub).")
    ap.add_argument("--folds", nargs="+", default=["0", "1", "2", "3", "4"],
                    help="Folds to ensemble (default: all 5).")
    ap.add_argument("--postprocess", action="store_true",
                    help="Also apply the cross-validated postprocessing.")
    ap.add_argument("--device", default=None, choices=["cuda", "cpu", "mps"])
    ap.add_argument("--disable-tta", action="store_true",
                    help="Skip mirroring TTA; much faster, slightly worse.")
    args = ap.parse_args()

    cases = check_input(args.input)
    results_root = resolve_weights(args.weights)

    env = dict(os.environ)
    env["nnUNet_results"] = results_root
    # nnU-Net insists these exist even for pure inference.
    env.setdefault("nnUNet_raw", os.path.join(args.output, "_nnunet_raw"))
    env.setdefault("nnUNet_preprocessed", os.path.join(args.output, "_nnunet_pre"))
    os.makedirs(env["nnUNet_raw"], exist_ok=True)
    os.makedirs(env["nnUNet_preprocessed"], exist_ok=True)
    os.makedirs(args.output, exist_ok=True)

    cmd = ["nnUNetv2_predict",
           "-d", DATASET, "-i", args.input, "-o", args.output,
           "-f", *args.folds,
           "-tr", "nnUNetTrainer", "-c", "3d_fullres", "-p", "nnUNetPlans"]
    if args.device:
        cmd += ["-device", args.device]
    if args.disable_tta:
        cmd += ["--disable_tta"]
    run(cmd, env)

    final_dir = args.output
    if args.postprocess:
        pp_dir = args.output.rstrip("/") + "_pp"
        model_dir = os.path.join(results_root, DATASET, TRAINER_DIR)
        run(["nnUNetv2_apply_postprocessing",
             "-i", args.output, "-o", pp_dir,
             "-pp_pkl_file", os.path.join(model_dir, "postprocessing.pkl"),
             "-plans_json", os.path.join(model_dir, "plans.json")], env)
        final_dir = pp_dir

    logging.info(f"Wrote {final_dir}")
    summarise(final_dir, cases)


if __name__ == "__main__":
    main()
