#!/usr/bin/env python3
"""
Regenerate npz + npy caches with correct normalization (0-255 gray level).

The previous caches were created with clip_max=100 (wrong — data is 0-255 gray level).
This script deletes and rebuilds both caches with clip_max=255.

Usage:
    python regenerate_cache.py
    # Or via SLURM:  sbatch --wrap="python regenerate_cache.py" --time=2:00:00 -p gpu
"""

import os
import sys
import shutil
import time
import glob
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from Data.dataset import _nifti_to_npz, _ensure_npy

DERIVATIVES_DIR = os.environ.get("SUS_CURATED_ROOT", "/path/to/Curated-SUS-2025")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "Data")
NPZ_DIR = os.path.join(CACHE_DIR, "npz_cache")
NPY_DIR = os.path.join(CACHE_DIR, "npy_cache")

CLIP_MIN = 0.0
CLIP_MAX = 255.0  # Correct: gray-level range


def discover_patients(derivatives_dir: str):
    """Find all patient CTP + label pairs."""
    import re
    deriv = os.path.join(derivatives_dir, "derivatives")
    patients = []
    for subj in sorted(os.listdir(deriv)):
        if not re.match(r"sub-stroke_\d{2}_\d{3}$", subj):
            continue
        ses_dir = os.path.join(deriv, subj, "ses-01")
        ctp = os.path.join(ses_dir, f"{subj}_ses-01_space-ncct_ctp.nii.gz")
        lbl = os.path.join(ses_dir, f"{subj}_ses-01_space-ncct_ctp_lesion_msk_3class.nii.gz")
        if os.path.isfile(ctp) and os.path.isfile(lbl):
            patients.append((subj, ctp, lbl))
    return patients


def main():
    patients = discover_patients(DERIVATIVES_DIR)
    print(f"Found {len(patients)} patients")

    # --- Delete old caches ---
    for d, name in [(NPZ_DIR, "npz_cache"), (NPY_DIR, "npy_cache")]:
        if os.path.isdir(d):
            size_gb = sum(
                os.path.getsize(os.path.join(d, f))
                for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))
            ) / 1e9
            print(f"Deleting old {name}/ ({size_gb:.1f} GB)...")
            shutil.rmtree(d)
    
    os.makedirs(NPZ_DIR, exist_ok=True)
    os.makedirs(NPY_DIR, exist_ok=True)

    # --- Stage 1: NIfTI → compressed .npz (with correct normalization) ---
    print(f"\n{'='*60}")
    print(f"  Stage 1: NIfTI → .npz (clip_min={CLIP_MIN}, clip_max={CLIP_MAX})")
    print(f"{'='*60}")
    t_start = time.time()

    npz_paths = []
    for i, (pid, ctp, lbl) in enumerate(patients):
        basename = os.path.basename(ctp).replace('.nii.gz', '').replace('.nii', '')
        npz_path = os.path.join(NPZ_DIR, f"{basename}.npz")
        
        print(f"  [{i+1:3d}/{len(patients)}] {pid}...", end=" ", flush=True)
        t0 = time.time()
        _nifti_to_npz(ctp, lbl, npz_path, clip_min=CLIP_MIN, clip_max=CLIP_MAX)
        sz = os.path.getsize(npz_path) / 1e6
        print(f"done in {time.time()-t0:.1f}s ({sz:.0f} MB)")
        npz_paths.append(npz_path)

    npz_total = sum(os.path.getsize(p) for p in npz_paths) / 1e9
    print(f"\nStage 1 complete: {len(npz_paths)} files, {npz_total:.1f} GB, "
          f"{time.time()-t_start:.0f}s total")

    # --- Stage 2: .npz → uncompressed .npy (valid slices only) ---
    print(f"\n{'='*60}")
    print(f"  Stage 2: .npz → .npy (mmap-capable)")
    print(f"{'='*60}")
    t_start = time.time()

    for i, npz_path in enumerate(npz_paths):
        pid = patients[i][0]
        print(f"  [{i+1:3d}/{len(npz_paths)}] {pid}...", end=" ", flush=True)
        t0 = time.time()
        ctp_npy, lbl_npy, vs_npy = _ensure_npy(npz_path, NPY_DIR)
        print(f"  {time.time()-t0:.1f}s")

    npy_total = sum(
        os.path.getsize(os.path.join(NPY_DIR, f))
        for f in os.listdir(NPY_DIR)
    ) / 1e9
    print(f"\nStage 2 complete: {npy_total:.1f} GB, {time.time()-t_start:.0f}s total")

    # --- Verify ---
    print(f"\n{'='*60}")
    print(f"  Verification")
    print(f"{'='*60}")
    # Check a few volumes
    for npz_path in npz_paths[:3]:
        basename = os.path.basename(npz_path).replace('.npz', '')
        ctp_npy = os.path.join(NPY_DIR, f"{basename}_ctp.npy")
        ctp = np.load(ctp_npy, mmap_mode='r')
        sl = ctp[:, 0, :256, :256]
        print(f"  {basename}: shape={ctp.shape}, min={sl.min():.3f}, max={sl.max():.3f}")
        assert sl.max() <= 1.001, f"ERROR: max value {sl.max()} > 1.0 — normalization still wrong!"
    
    print("\nAll checks passed! Caches regenerated with clip_max=255.")


if __name__ == "__main__":
    main()
