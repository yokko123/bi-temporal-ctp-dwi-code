"""
Dataset classes for loading 4D CTP data for mJNet training.

Data format (NIfTI on disk):
  - CTP:   (X, Y, Z, T) e.g. (512, 512, 52, 40)
  - Label: (X, Y, Z)    e.g. (512, 512, 52) with values {0,1,2,3}

Internal working format (after loading via SimpleITK):
  - CTP:   (T, Z, Y, X)
  - Label: (Z, Y, X)

Workflow:
  1. First run: reads NIfTI (slow ~16s/vol), saves .npz to cache_dir
  2. Second run: converts .npz → uncompressed .npy (valid slices only, ~8s/vol)
  3. All subsequent runs: memory-maps .npy files (~0.01s/vol, near-instant)

CTP is registered to NCCT space, so many Z-slices have zero CTP signal.
Only slices with actual CTP data are stored and used for training.

Cache layout (per volume):
  npz_cache/{name}.npz              — compressed archive (backward compat)
  npy_cache/{name}_ctp.npy          — (T, Zv, Y, X) float32, uncompressed
  npy_cache/{name}_lbl.npy          — (Zv, Y, X)    int8,    uncompressed
  npy_cache/{name}_valid_slices.npy — (Zv,)          int32   (original z-indices)
"""

import os
import glob
import numpy as np
import SimpleITK as sitk
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict
import random
import time


# ======================================================================
#  Helpers: NIfTI -> npz -> npy conversion (one-time cost per volume)
# ======================================================================

def _nifti_to_npz(ctp_nifti: str, label_nifti: str, npz_path: str,
                   clip_min: float = 0.0, clip_max: float = 255.0) -> None:
    """Convert one CTP+label NIfTI pair to a single .npz file."""
    # CTP: SimpleITK gives (T, Z, Y, X) for 4D
    ctp = sitk.GetArrayFromImage(sitk.ReadImage(ctp_nifti)).astype(np.float32)
    ctp = (ctp - clip_min) / (clip_max - clip_min + 1e-8)

    # Label: SimpleITK gives (Z, Y, X) for 3D
    lbl = sitk.GetArrayFromImage(sitk.ReadImage(label_nifti)).astype(np.int64)

    # Valid slices mask (where CTP has signal)
    slice_sums = np.sum(np.abs(ctp), axis=(0, 2, 3))  # (Z,)
    valid_slices = np.where(slice_sums > 0)[0].astype(np.int32)

    # Use compressed format for faster I/O (better than gzip on network FS)
    np.savez_compressed(npz_path, ctp=ctp, label=lbl, valid_slices=valid_slices)


def _ensure_npz(ctp_path: str, label_path: str, cache_dir: str,
                clip_min: float = 0.0, clip_max: float = 255.0) -> str:
    """Return path to .npz file, creating it from NIfTI if needed."""
    basename = os.path.basename(ctp_path).replace('.nii.gz', '').replace('.nii', '')
    npz_path = os.path.join(cache_dir, f"{basename}.npz")
    if not os.path.exists(npz_path):
        print(f"    Converting {os.path.basename(ctp_path)} -> .npz (first time, slow)...")
        t0 = time.time()
        _nifti_to_npz(ctp_path, label_path, npz_path, clip_min, clip_max)
        print(f"    Done in {time.time() - t0:.1f}s  ({os.path.getsize(npz_path) / 1e6:.0f} MB)")
    return npz_path


def _ensure_npy(npz_path: str, npy_dir: str) -> Tuple[str, str, str]:
    """Convert compressed .npz → uncompressed .npy files (valid slices only).

    Returns (ctp_npy_path, lbl_npy_path, valid_slices_npy_path).

    Each .npy file is uncompressed, which:
      - Supports memory-mapping (mmap_mode='r') → near-instant loading
      - Eliminates gzip decompression (~8s/vol → ~0.01s/vol)
      - Stores only valid Z-slices → ~50% less data than full volume
    """
    basename = os.path.basename(npz_path).replace('.npz', '')
    ctp_npy = os.path.join(npy_dir, f"{basename}_ctp.npy")
    lbl_npy = os.path.join(npy_dir, f"{basename}_lbl.npy")
    vs_npy  = os.path.join(npy_dir, f"{basename}_valid_slices.npy")

    if os.path.exists(ctp_npy) and os.path.exists(lbl_npy) and os.path.exists(vs_npy):
        return ctp_npy, lbl_npy, vs_npy

    # One-time conversion: decompress .npz → save as .npy
    print(f"    Converting {basename}.npz -> .npy (one-time)...", end=" ", flush=True)
    t0 = time.time()
    data = np.load(npz_path, allow_pickle=False)
    ctp = data['ctp']             # (T, Z, Y, X) float32
    lbl = data['label']           # (Z, Y, X)
    vs  = data['valid_slices']    # (Zv,) int32
    data.close()

    # Save only valid slices, uncompressed — supports mmap
    np.save(ctp_npy, ctp[:, vs, :, :].astype(np.float32))  # (T, Zv, Y, X)
    np.save(lbl_npy, lbl[vs, :, :].astype(np.int8))         # (Zv, Y, X) int8 saves 8×
    np.save(vs_npy,  vs.astype(np.int32))                    # original z-indices

    ctp_mb = os.path.getsize(ctp_npy) / 1e6
    lbl_mb = os.path.getsize(lbl_npy) / 1e6
    print(f"done in {time.time() - t0:.1f}s  (ctp: {ctp_mb:.0f}MB, lbl: {lbl_mb:.0f}MB)")

    return ctp_npy, lbl_npy, vs_npy


# ======================================================================
#  Training dataset — memory-mapped .npy files
# ======================================================================

class CTPPatchDataset(Dataset):
    """
    Patch-based dataset for mJNet training.

    Two-stage cache:
      1. .npz (compressed, from NIfTI)  — created on first ever run
      2. .npy (uncompressed, valid slices only) — created from .npz, supports mmap

    CTP volumes are memory-mapped (mmap_mode='r'):
      - Near-instant initialization (~0.01s/vol vs ~8s/vol for compressed .npz)
      - Near-zero RAM footprint (OS pages data in on demand)
      - Shared across DataLoader worker processes via OS page cache

    Labels are loaded into RAM as int8 (~6.5 MB/vol, ~780 MB for 120 vols):
      - Needed for patch index building (brain fraction + lesion checks)
      - Tiny compared to CTP volumes

    Supports oversampling of rare classes (penumbra/core) to combat
    extreme class imbalance. When oversample_lesion > 1, patches containing
    core or penumbra voxels are duplicated in the index.

    Returns:
        ctp_patch:   (T, patch_size, patch_size)  float32  [0, 1]
        label_patch: (patch_size, patch_size)      int64    {0,1,2,3}
    """

    def __init__(
        self,
        ctp_paths: List[str],
        label_paths: List[str],
        cache_dir: str,
        patch_size: int = 16,
        stride: int = 4,
        num_classes: int = 4,
        augment: bool = True,
        skip_empty_patches: bool = True,
        min_brain_fraction: float = 0.1,
        clip_min: float = 0.0,
        clip_max: float = 255.0,
        oversample_lesion: int = 1,
    ):
        self.patch_size = patch_size
        self.stride = stride
        self.num_classes = num_classes
        self.augment = augment
        self.skip_empty_patches = skip_empty_patches
        self.min_brain_fraction = min_brain_fraction
        self.oversample_lesion = max(1, oversample_lesion)

        # Store original NIfTI paths (needed for prediction visualisations)
        self.ctp_paths: List[str] = list(ctp_paths)
        self.label_paths: List[str] = list(label_paths)

        # ---- Stage 1: Ensure compressed .npz cache exists ----
        npz_dir = os.path.join(cache_dir, "npz_cache") if "npz_cache" not in cache_dir else cache_dir
        os.makedirs(npz_dir, exist_ok=True)

        print(f"Preparing {len(ctp_paths)} volumes...")
        t0 = time.time()
        npz_paths: List[str] = []
        for ctp_p, lbl_p in zip(ctp_paths, label_paths):
            npz = _ensure_npz(ctp_p, lbl_p, npz_dir, clip_min, clip_max)
            npz_paths.append(npz)
        print(f"  All .npz files ready in {time.time() - t0:.1f}s")

        # ---- Stage 2: Ensure uncompressed .npy cache exists ----
        npy_dir = npz_dir.replace("npz_cache", "npy_cache")
        os.makedirs(npy_dir, exist_ok=True)

        print(f"Setting up memory-mapped volumes ({npy_dir})...")
        t_setup = time.time()

        self._mmap_ctp: Dict[int, np.ndarray] = {}      # vol_idx -> mmap (T, Zv, Y, X)
        self._vol_lbl:  Dict[int, np.ndarray] = {}       # vol_idx -> (Zv, Y, X) int8 in RAM
        self._valid_slices: Dict[int, np.ndarray] = {}   # vol_idx -> (Zv,) int32
        self._npy_paths: Dict[int, Tuple[str, str, str]] = {}  # for reference

        for idx, npz_path in enumerate(npz_paths):
            ctp_npy, lbl_npy, vs_npy = _ensure_npy(npz_path, npy_dir)

            self._mmap_ctp[idx] = np.load(ctp_npy, mmap_mode='r')    # memory-mapped!
            self._vol_lbl[idx]  = np.load(lbl_npy)                    # in RAM (small)
            self._valid_slices[idx] = np.load(vs_npy)
            self._npy_paths[idx] = (ctp_npy, lbl_npy, vs_npy)

        elapsed = time.time() - t_setup
        total_lbl_mb = sum(a.nbytes for a in self._vol_lbl.values()) / 1e6
        n_vols = len(npz_paths)
        print(f"  {n_vols} volumes ready in {elapsed:.1f}s "
              f"(labels in RAM: {total_lbl_mb:.0f} MB, CTP: memory-mapped)\n")

        # Build patch index: [(vol_idx, z_local, y, x), ...]
        self.patches: List[Tuple[int, int, int, int]] = []
        self._build_patch_index()

    # ------------------------------------------------------------------
    # Patch index — uses local z-indices (valid slices only)
    # ------------------------------------------------------------------
    def _build_patch_index(self):
        print("Building patch index...")
        t0 = time.time()
        total_slices_used = 0

        ps = self.patch_size
        st = self.stride

        base_patches = []       # All valid patches
        lesion_patches = []     # Patches containing penumbra (class 2) or core (class 3)

        for idx in range(len(self._vol_lbl)):
            # Labels already contain ONLY valid slices (stored as int8)
            lbl = self._vol_lbl[idx]      # (Zv, Y, X) int8
            Zv, Y, X = lbl.shape
            total_slices_used += Zv

            ys = np.arange(0, Y - ps + 1, st)
            xs = np.arange(0, X - ps + 1, st)

            if not self.skip_empty_patches:
                for z_local in range(Zv):
                    for y in ys:
                        for x in xs:
                            patch_coord = (idx, z_local, int(y), int(x))
                            base_patches.append(patch_coord)
                            # Check if patch contains penumbra or core
                            patch_lbl = lbl[z_local, y:y+ps, x:x+ps]
                            if np.any((patch_lbl == 2) | (patch_lbl == 3)):
                                lesion_patches.append(patch_coord)
            else:
                threshold = self.min_brain_fraction * (ps * ps)
                for z_local in range(Zv):
                    brain_mask = (lbl[z_local] > 0).astype(np.float64)
                    integral = np.cumsum(np.cumsum(brain_mask, axis=0), axis=1)
                    S = np.zeros((Y + 1, X + 1), dtype=np.float64)
                    S[1:, 1:] = integral

                    yy = ys[:, None]
                    xx = xs[None, :]
                    patch_sums = (S[yy + ps, xx + ps] - S[yy, xx + ps]
                                  - S[yy + ps, xx] + S[yy, xx])

                    valid_ys, valid_xs = np.where(patch_sums >= threshold)
                    for i in range(len(valid_ys)):
                        y_coord = int(ys[valid_ys[i]])
                        x_coord = int(xs[valid_xs[i]])
                        patch_coord = (idx, z_local, y_coord, x_coord)
                        base_patches.append(patch_coord)
                        # Check if patch contains penumbra (class 2) or core (class 3)
                        patch_lbl = lbl[z_local, y_coord:y_coord+ps, x_coord:x_coord+ps]
                        if np.any((patch_lbl == 2) | (patch_lbl == 3)):
                            lesion_patches.append(patch_coord)

        # Build final patch list with oversampling
        self.patches = list(base_patches)
        if self.oversample_lesion > 1 and len(lesion_patches) > 0:
            # Add extra copies of lesion patches (penumbra + core)
            extra_copies = self.oversample_lesion - 1
            self.patches.extend(lesion_patches * extra_copies)

        print(f"  Slices used: {total_slices_used} (all valid, no empty CTP slices stored)")
        print(f"  Base patches: {len(base_patches):,}")
        print(f"  Lesion patches (pen+core): {len(lesion_patches):,} ({100*len(lesion_patches)/max(len(base_patches),1):.1f}%)")
        if self.oversample_lesion > 1:
            print(f"  Oversampling: {self.oversample_lesion}x -> added {len(lesion_patches) * (self.oversample_lesion - 1):,} extra lesion patches")
        print(f"  Total patches (after oversampling): {len(self.patches):,}")
        print(f"  Patch index built in {time.time() - t0:.1f}s\n")

    # ------------------------------------------------------------------
    # Augmentation (pure PyTorch — no numpy/scipy in the hot path)
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_augmentation(ctp_patch, label_patch, aug_idx):
        """Geometric augmentation on torch tensors.
        aug_idx: 0=none, 1=rot90, 2=rot180, 3=rot270, 4=flipud, 5=fliplr.
        ctp_patch: (T, H, W) float32,  label_patch: (H, W) int64.
        """
        if aug_idx == 1:
            ctp_patch = torch.rot90(ctp_patch, k=1, dims=(1, 2))
            label_patch = torch.rot90(label_patch, k=1, dims=(0, 1))
        elif aug_idx == 2:
            ctp_patch = torch.rot90(ctp_patch, k=2, dims=(1, 2))
            label_patch = torch.rot90(label_patch, k=2, dims=(0, 1))
        elif aug_idx == 3:
            ctp_patch = torch.rot90(ctp_patch, k=3, dims=(1, 2))
            label_patch = torch.rot90(label_patch, k=3, dims=(0, 1))
        elif aug_idx == 4:
            ctp_patch = torch.flip(ctp_patch, dims=[1])
            label_patch = torch.flip(label_patch, dims=[0])
        elif aug_idx == 5:
            ctp_patch = torch.flip(ctp_patch, dims=[2])
            label_patch = torch.flip(label_patch, dims=[1])
        return ctp_patch, label_patch

    @staticmethod
    def _apply_intensity_augmentation(ctp_patch):
        """Stochastic intensity augmentation using pure PyTorch ops.

        ctp_patch: (T, H, W) float32 tensor.
        Randomly applies (each with 50% probability):
          - Additive Gaussian noise  (std ~ U[0.005, 0.03])
          - Multiplicative brightness (factor ~ U[0.85, 1.15])
          - Gaussian blur via torchvision (kernel 3 or 5, sigma ~ U[0.3, 1.0])
        """
        # Additive Gaussian noise
        if random.random() < 0.5:
            std = random.uniform(0.005, 0.03)
            ctp_patch = ctp_patch + torch.randn_like(ctp_patch) * std

        # Multiplicative brightness
        if random.random() < 0.5:
            factor = random.uniform(0.85, 1.15)
            ctp_patch = ctp_patch * factor

        # Gaussian blur — torchvision treats (T, H, W) as (C, H, W) and blurs H,W
        if random.random() < 0.5:
            kernel_size = random.choice([3, 5])
            sigma = random.uniform(0.3, 1.0)
            ctp_patch = TF.gaussian_blur(ctp_patch, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])

        return ctp_patch.clamp_(0.0, 1.0)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        vol_idx, z_local, y, x = self.patches[idx]
        ps = self.patch_size

        # CTP: memory-mapped read — only pages in the patch data (~40 KB)
        ctp_patch = torch.from_numpy(
            self._mmap_ctp[vol_idx][:, z_local, y:y+ps, x:x+ps].copy()
        )  # (T, ps, ps) float32

        # Label: from RAM (int8) → convert to int64 for loss function
        lbl_patch = torch.from_numpy(
            self._vol_lbl[vol_idx][z_local, y:y+ps, x:x+ps].copy()
        ).long()  # (ps, ps) int64

        # Fully stochastic augmentation — random transform each time
        if self.augment:
            aug_idx = random.randint(0, 5)  # 0=none, 1-5=transforms
            if aug_idx > 0:
                ctp_patch, lbl_patch = self._apply_augmentation(ctp_patch, lbl_patch, aug_idx)
            ctp_patch = self._apply_intensity_augmentation(ctp_patch)
        else:
            # Safety clamp for validation (ensure [0,1] even without augmentation)
            ctp_patch = ctp_patch.clamp_(0.0, 1.0)

        return ctp_patch, lbl_patch


# ======================================================================
#  Volume-level dataset (for inference / testing)
# ======================================================================
class CTPVolumeDataset(Dataset):
    """
    Returns one Z-slice at a time with all its patches.
    Only returns slices that have CTP data.
    Prefers .npy cache (mmap) > .npz cache > direct NIfTI loading.
    """

    def __init__(
        self,
        ctp_paths: List[str],
        label_paths: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
        patch_size: int = 16,
        stride: int = 4,
        clip_min: float = 0.0,
        clip_max: float = 255.0,
    ):
        self.ctp_paths = ctp_paths
        self.label_paths = label_paths
        self.patch_size = patch_size
        self.stride = stride

        # Resolve cache paths: prefer npy > npz > NIfTI
        self._npy_ctp: Dict[int, Optional[np.ndarray]] = {}    # mmap or None
        self._npy_lbl: Dict[int, Optional[np.ndarray]] = {}    # mmap or None
        self._npy_vs:  Dict[int, Optional[np.ndarray]] = {}    # valid_slices or None
        self.npz_paths: List[Optional[str]] = [None] * len(ctp_paths)

        if cache_dir is not None:
            npz_dir = os.path.join(cache_dir, "npz_cache") if "npz_cache" not in cache_dir else cache_dir
            npy_dir = npz_dir.replace("npz_cache", "npy_cache")

            for i, cp in enumerate(ctp_paths):
                bn = os.path.basename(cp).replace('.nii.gz', '').replace('.nii', '')

                # Try npy first (fastest)
                ctp_npy = os.path.join(npy_dir, f"{bn}_ctp.npy")
                lbl_npy = os.path.join(npy_dir, f"{bn}_lbl.npy")
                vs_npy  = os.path.join(npy_dir, f"{bn}_valid_slices.npy")

                if os.path.exists(ctp_npy) and os.path.exists(lbl_npy) and os.path.exists(vs_npy):
                    self._npy_ctp[i] = np.load(ctp_npy, mmap_mode='r')
                    self._npy_lbl[i] = np.load(lbl_npy, mmap_mode='r')
                    self._npy_vs[i]  = np.load(vs_npy)
                else:
                    # Fallback to npz
                    npz = os.path.join(npz_dir, f"{bn}.npz")
                    self.npz_paths[i] = npz if os.path.exists(npz) else None

        self.clip_min = clip_min
        self.clip_max = clip_max

        # Build slice index: [(vol_idx, z_local_or_orig, is_npy), ...]
        self.slices: List[Tuple[int, int]] = []
        self._slice_is_npy: List[bool] = []
        self._build_slice_index()

    def _load_ctp_from_nifti(self, path: str) -> np.ndarray:
        data = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)
        data = (data - self.clip_min) / (self.clip_max - self.clip_min + 1e-8)
        return data  # (T,Z,Y,X)

    def _build_slice_index(self):
        for vol_idx in range(len(self.ctp_paths)):
            if vol_idx in self._npy_ctp and self._npy_ctp[vol_idx] is not None:
                # npy cache: local z-indices, map back to original via valid_slices
                vs = self._npy_vs[vol_idx]
                Zv = len(vs)
                for z_local in range(Zv):
                    self.slices.append((vol_idx, z_local))
                    self._slice_is_npy.append(True)
                print(f"  Volume {vol_idx}: {Zv} valid slices (npy-mmap)")
            elif self.npz_paths[vol_idx] is not None:
                data = np.load(self.npz_paths[vol_idx], allow_pickle=False)
                valid_z = data['valid_slices']
                Z = data['label'].shape[0]
                data.close()
                for z in valid_z:
                    self.slices.append((vol_idx, int(z)))
                    self._slice_is_npy.append(False)
                print(f"  Volume {vol_idx}: {len(valid_z)}/{Z} valid slices (npz)")
            else:
                ctp = self._load_ctp_from_nifti(self.ctp_paths[vol_idx])
                slice_sums = np.sum(np.abs(ctp), axis=(0, 2, 3))
                valid_z = np.where(slice_sums > 0)[0]
                Z = ctp.shape[1]
                for z in valid_z:
                    self.slices.append((vol_idx, int(z)))
                    self._slice_is_npy.append(False)
                print(f"  Volume {vol_idx}: {len(valid_z)}/{Z} valid slices (nifti)")

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        vol_idx, z = self.slices[idx]
        is_npy = self._slice_is_npy[idx]

        if is_npy:
            # Fast path: memory-mapped .npy
            slice_ctp = np.array(self._npy_ctp[vol_idx][:, z, :, :])   # (T, Y, X)
            lbl_full  = np.array(self._npy_lbl[vol_idx][z, :, :])       # (Y, X)
            # Map local z back to original z-index for metadata
            orig_z = int(self._npy_vs[vol_idx][z])
        elif self.npz_paths[vol_idx] is not None:
            data = np.load(self.npz_paths[vol_idx], mmap_mode='r')
            slice_ctp = np.array(data['ctp'][:, z, :, :])   # (T, Y, X)
            lbl_full = np.array(data['label'][z])             # (Y, X)
            data.close()
            orig_z = z
        else:
            ctp = self._load_ctp_from_nifti(self.ctp_paths[vol_idx])
            slice_ctp = ctp[:, z, :, :]
            lbl_full = None
            if self.label_paths is not None:
                lbl = sitk.GetArrayFromImage(sitk.ReadImage(self.label_paths[vol_idx])).astype(np.int64)
                lbl_full = lbl[z]
            orig_z = z

        _, Y, X = slice_ctp.shape
        ps, st = self.patch_size, self.stride

        patches, coords = [], []
        for y in range(0, Y - ps + 1, st):
            for x in range(0, X - ps + 1, st):
                patches.append(slice_ctp[:, y:y+ps, x:x+ps])
                coords.append((y, x))

        patches = np.stack(patches)

        result = {
            'patches': torch.from_numpy(patches).float(),
            'coords': coords,
            'slice_shape': (Y, X),
            'vol_idx': vol_idx,
            'slice_idx': orig_z,
            'ctp_path': self.ctp_paths[vol_idx],
        }

        if lbl_full is not None:
            result['label'] = torch.from_numpy(lbl_full.astype(np.int64)).long()
        elif self.label_paths is not None:
            lbl = sitk.GetArrayFromImage(sitk.ReadImage(self.label_paths[vol_idx])).astype(np.int64)
            result['label'] = torch.from_numpy(lbl[z]).long()

        return result


# ======================================================================
#  Factory helpers
# ======================================================================
def _patient_id(path: str) -> str:
    """Extract patient ID from a NIfTI filename.

    Handles both curated (``sub-stroke_XX_XXX_ses-01_...``) and
    legacy (``sub-01_001_...``) naming conventions.
    """
    import re
    basename = os.path.basename(path)
    m = re.match(r"(sub-stroke_\d{2}_\d{3})", basename)
    if m:
        return m.group(1)
    parts = basename.replace('.nii.gz', '').replace('.nii', '').split('_')
    return '_'.join(parts[:2])


def _discover_pairs_from_derivatives(derivatives_dir: str) -> list:
    """
    Discover CTP + 3-class label pairs from the curated BIDS-like layout:

        derivatives/
          sub-stroke_XX_XXX/
            ses-01/
              {pid}_ses-01_space-ncct_ctp.nii.gz
              {pid}_ses-01_space-ncct_ctp_lesion_msk_3class.nii.gz

    Returns a list of (ctp_path, label_path) tuples.
    """
    import re
    pairs = []
    deriv_root = os.path.join(derivatives_dir, "derivatives")
    if not os.path.isdir(deriv_root):
        raise FileNotFoundError(f"No 'derivatives' folder inside {derivatives_dir}")

    for subj in sorted(os.listdir(deriv_root)):
        if not re.match(r"sub-stroke_\d{2}_\d{3}$", subj):
            continue
        ses_dir = os.path.join(deriv_root, subj, "ses-01")
        if not os.path.isdir(ses_dir):
            continue

        ctp_path = os.path.join(ses_dir, f"{subj}_ses-01_space-ncct_ctp.nii.gz")
        label_path = os.path.join(ses_dir, f"{subj}_ses-01_space-ncct_ctp_lesion_msk_3class.nii.gz")

        if os.path.isfile(ctp_path) and os.path.isfile(label_path):
            pairs.append((ctp_path, label_path))
        else:
            missing = []
            if not os.path.isfile(ctp_path):
                missing.append("CTP")
            if not os.path.isfile(label_path):
                missing.append("3-class label")
            print(f"  Warning: {subj} missing {', '.join(missing)} — skipped")

    return pairs


# ======================================================================
# Internal helpers (pair discovery, loader construction)
# ======================================================================

def _discover_and_filter_pairs(
    config,
    derivatives_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
    label_dir: Optional[str] = None,
    test_patients: Optional[List[str]] = None,
    num_patients: Optional[int] = None,
) -> List[Tuple[str, str]]:
    """Discover CTP-label pairs, exclude test patients, optionally limit count.

    Returns a deterministically-shuffled list of ``(ctp_path, label_path)`` tuples.
    """
    data_cfg = config.data if hasattr(config, 'data') else config

    if test_patients is None:
        test_patients = getattr(data_cfg, 'test_patients', [])

    # ---- discover pairs ----
    if derivatives_dir is not None:
        print(f"Using curated dataset at: {derivatives_dir}")
        matched_pairs = _discover_pairs_from_derivatives(derivatives_dir)
        print(f"Found {len(matched_pairs)} CTP-label pairs from curated dataset")
    elif data_dir is not None and label_dir is not None:
        ctp_files = sorted(glob.glob(os.path.join(data_dir, "*.nii*")))
        label_files = sorted(glob.glob(os.path.join(label_dir, "*.nii*")))
        print(f"Found {len(ctp_files)} CTP files and {len(label_files)} label files")

        matched_pairs = []
        for ctp_path in ctp_files:
            ctp_parts = os.path.basename(ctp_path).replace('.nii.gz', '').replace('.nii', '').split('_')
            for label_path in label_files:
                label_parts = os.path.basename(label_path).replace('.nii.gz', '').replace('.nii', '').split('_')
                if (len(ctp_parts) >= 2 and len(label_parts) >= 2
                        and ctp_parts[0] == label_parts[0]
                        and ctp_parts[1] == label_parts[1]):
                    matched_pairs.append((ctp_path, label_path))
                    break

        print(f"Matched {len(matched_pairs)} CTP-label pairs")
    else:
        raise ValueError(
            "Provide either --derivatives_dir (curated dataset) "
            "or both --data_dir and --label_dir (legacy flat dirs)."
        )

    if len(matched_pairs) == 0:
        raise ValueError("No matching CTP-label pairs found!")

    # ---- exclude test patients ----
    if test_patients:
        excluded_pairs, kept_pairs = [], []
        for pair in matched_pairs:
            ctp_basename = os.path.basename(pair[0])
            is_test = any(test_id in ctp_basename for test_id in test_patients)
            (excluded_pairs if is_test else kept_pairs).append(pair)

        print(f"\n🔒 Test set (excluded from training): {len(excluded_pairs)} patients")
        for p in excluded_pairs[:5]:
            print(f"    - {os.path.basename(p[0])}")
        if len(excluded_pairs) > 5:
            print(f"    ... and {len(excluded_pairs) - 5} more")

        matched_pairs = kept_pairs
        print(f"📊 Remaining for train/val: {len(matched_pairs)} patients\n")

    # ---- limit (for quick tests) ----
    if num_patients is not None and num_patients < len(matched_pairs):
        print(f"Limiting to {num_patients} patients (out of {len(matched_pairs)})")
        matched_pairs = matched_pairs[:num_patients]

    # ---- deterministic shuffle ----
    seed = getattr(data_cfg, 'random_seed', getattr(data_cfg, 'seed', 42))
    random.seed(seed)
    random.shuffle(matched_pairs)

    return matched_pairs


def _build_loaders(
    train_pairs: List[Tuple[str, str]],
    val_pairs: List[Tuple[str, str]],
    config,
    train_config=None,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and val DataLoaders from pre-split (ctp, label) pair lists."""
    if hasattr(config, 'data'):
        data_cfg = config.data
        train_cfg = config.train
    else:
        data_cfg = config
        train_cfg = train_config if train_config is not None else config

    train_ctp    = [p[0] for p in train_pairs]
    train_labels = [p[1] for p in train_pairs]
    val_ctp      = [p[0] for p in val_pairs]
    val_labels   = [p[1] for p in val_pairs]

    # ---- verify no patient overlap ----
    train_ids = set(_patient_id(p) for p in train_ctp)
    val_ids   = set(_patient_id(p) for p in val_ctp)
    overlap   = train_ids & val_ids
    assert len(overlap) == 0, f"DATA LEAK: patients in both train & val: {overlap}"
    print(f"✓ No patient overlap — train: {len(train_ids)}, val: {len(val_ids)}")

    print(f"Training volumes: {len(train_ctp)}")
    print(f"Validation volumes: {len(val_ctp)}")

    # ---- config values with fallbacks ----
    patch_size     = getattr(data_cfg, 'patch_size', 16)
    stride         = getattr(data_cfg, 'stride', 4)
    num_classes    = getattr(data_cfg, 'num_classes', 4)
    clip_min       = getattr(data_cfg, 'clip_min', 0.0)
    clip_max       = getattr(data_cfg, 'clip_max', 255.0)
    do_augment     = getattr(train_cfg, 'augmentation', True)
    batch_size     = getattr(train_cfg, 'batch_size', 32)
    pin_memory     = getattr(train_cfg, 'pin_memory', True)
    num_workers    = getattr(train_cfg, 'num_workers', 8)
    cache_dir      = getattr(data_cfg, 'cache_dir', None)

    if cache_dir is None:
        raise ValueError("cache_dir must be set in config for lazy-loading dataset")

    oversample_lesion = getattr(data_cfg, 'oversample_lesion', 1)

    # ---- build datasets ----
    train_dataset = CTPPatchDataset(
        ctp_paths=train_ctp,
        label_paths=train_labels,
        cache_dir=cache_dir,
        patch_size=patch_size,
        stride=stride,
        num_classes=num_classes,
        augment=do_augment,
        skip_empty_patches=True,
        clip_min=clip_min,
        clip_max=clip_max,
        oversample_lesion=oversample_lesion,
    )
    val_dataset = CTPPatchDataset(
        ctp_paths=val_ctp,
        label_paths=val_labels,
        cache_dir=cache_dir,
        patch_size=patch_size,
        stride=stride,
        num_classes=num_classes,
        augment=False,
        skip_empty_patches=False,
        clip_min=clip_min,
        clip_max=clip_max,
    )

    # ---- build loaders ----
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    return train_loader, val_loader


# ======================================================================
# Public API
# ======================================================================

def get_data_loaders(
    config,
    derivatives_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
    label_dir: Optional[str] = None,
    train_config=None,
    train_patients: Optional[List[str]] = None,
    val_patients: Optional[List[str]] = None,
    test_patients: Optional[List[str]] = None,
    num_patients: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders (single train/val split).

    Supports two data source modes:
      1. **Curated (recommended):** Pass ``derivatives_dir``.
      2. **Legacy flat dirs:** Pass ``data_dir`` + ``label_dir``.

    Returns:
        (train_loader, val_loader)
    """
    data_cfg = config.data if hasattr(config, 'data') else config

    if train_patients is None or val_patients is None:
        # Auto-discover and split
        pairs = _discover_and_filter_pairs(
            config, derivatives_dir=derivatives_dir,
            data_dir=data_dir, label_dir=label_dir,
            test_patients=test_patients, num_patients=num_patients,
        )
        val_frac = getattr(data_cfg, 'val_split', 0.2)
        val_size = max(1, int(len(pairs) * val_frac))
        val_pairs = pairs[:val_size]
        train_pairs = pairs[val_size:]

        print(f"  Train patients ({len(train_pairs)}): {sorted([_patient_id(p[0]) for p in train_pairs])}")
        print(f"  Val   patients ({len(val_pairs)}): {sorted([_patient_id(p[0]) for p in val_pairs])}")
    else:
        # Explicit patient lists provided
        def _find(patients):
            cp, lp = [], []
            for pat in patients:
                if derivatives_dir is not None:
                    ses = os.path.join(derivatives_dir, "derivatives", pat, "ses-01")
                    cf = os.path.join(ses, f"{pat}_ses-01_space-ncct_ctp.nii.gz")
                    lf = os.path.join(ses, f"{pat}_ses-01_space-ncct_ctp_lesion_msk_3class.nii.gz")
                    if os.path.isfile(cf) and os.path.isfile(lf):
                        cp.append(cf); lp.append(lf)
                    else:
                        print(f"Warning: no files for patient {pat}")
                else:
                    cf = glob.glob(os.path.join(data_dir, f"*{pat}*.nii*"))
                    lf = glob.glob(os.path.join(label_dir, f"*{pat}*.nii*"))
                    if cf and lf:
                        cp.append(cf[0]); lp.append(lf[0])
                    else:
                        print(f"Warning: no files for patient {pat}")
            return list(zip(cp, lp))
        train_pairs = _find(train_patients)
        val_pairs = _find(val_patients)

    return _build_loaders(train_pairs, val_pairs, config, train_config)


def get_cv_fold_loaders(
    config,
    num_folds: int = 5,
    derivatives_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
    label_dir: Optional[str] = None,
    test_patients: Optional[List[str]] = None,
    num_patients: Optional[int] = None,
):
    """Generator yielding ``(fold_idx, train_loader, val_loader)`` for k-fold CV.

    Splitting is done at the **patient level** so no patient appears in both
    train and val within the same fold.

    Args:
        config:          Config object.
        num_folds:       Number of CV folds (default: 5).
        derivatives_dir: Curated dataset root.
        data_dir:        (Legacy) CTP directory.
        label_dir:       (Legacy) Label directory.
        test_patients:   Patient IDs to exclude entirely.
        num_patients:    Limit total patients (for quick tests).

    Yields:
        (fold_idx, train_loader, val_loader)
    """
    pairs = _discover_and_filter_pairs(
        config, derivatives_dir=derivatives_dir,
        data_dir=data_dir, label_dir=label_dir,
        test_patients=test_patients, num_patients=num_patients,
    )

    n = len(pairs)
    if n < num_folds:
        raise ValueError(f"Only {n} patients but requested {num_folds} folds")

    print(f"\n{'='*60}")
    print(f"  {num_folds}-fold cross-validation  ({n} patients)")
    print(f"{'='*60}\n")

    fold_size = n // num_folds
    for fold_idx in range(num_folds):
        val_start = fold_idx * fold_size
        val_end   = val_start + fold_size if fold_idx < num_folds - 1 else n
        val_pairs   = pairs[val_start:val_end]
        train_pairs = pairs[:val_start] + pairs[val_end:]

        train_ids = sorted([_patient_id(p[0]) for p in train_pairs])
        val_ids   = sorted([_patient_id(p[0]) for p in val_pairs])
        print(f"── Fold {fold_idx + 1}/{num_folds} ──  "
              f"train: {len(train_pairs)}, val: {len(val_pairs)}")
        print(f"   Train patients: {train_ids}")
        print(f"   Val   patients: {val_ids}")

        train_loader, val_loader = _build_loaders(train_pairs, val_pairs, config)
        yield fold_idx, train_loader, val_loader


def get_test_loader(
    test_patients: List[str],
    config,
    derivatives_dir: Optional[str] = None,
    ctp_dir: Optional[str] = None,
    label_dir: Optional[str] = None,
) -> DataLoader:
    """Create a test DataLoader (volume-level, one slice per item)."""
    data_cfg = config.data if hasattr(config, 'data') else config
    cache_dir = getattr(data_cfg, 'cache_dir', None)

    ctp_paths, label_paths_list = [], []
    for pat in test_patients:
        if derivatives_dir is not None:
            ses = os.path.join(derivatives_dir, "derivatives", pat, "ses-01")
            cf = os.path.join(ses, f"{pat}_ses-01_space-ncct_ctp.nii.gz")
            lf = os.path.join(ses, f"{pat}_ses-01_space-ncct_ctp_lesion_msk_3class.nii.gz")
            if os.path.isfile(cf):
                ctp_paths.append(cf)
                if os.path.isfile(lf):
                    label_paths_list.append(lf)
        else:
            cf = glob.glob(os.path.join(ctp_dir, f"*{pat}*.nii*"))
            if cf:
                ctp_paths.append(cf[0])
                if label_dir:
                    lf = glob.glob(os.path.join(label_dir, f"*{pat}*.nii*"))
                    if lf:
                        label_paths_list.append(lf[0])

    test_dataset = CTPVolumeDataset(
        ctp_paths=ctp_paths,
        label_paths=label_paths_list if label_paths_list else None,
        cache_dir=cache_dir,
        patch_size=getattr(data_cfg, 'patch_size', 16),
        stride=getattr(data_cfg, 'stride', 4),
        clip_min=getattr(data_cfg, 'clip_min', 0.0),
        clip_max=getattr(data_cfg, 'clip_max', 255.0),
    )

    return DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
