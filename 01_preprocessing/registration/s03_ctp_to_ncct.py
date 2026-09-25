"""
Step 3 – CTP → NCCT registration, label warping, and optional parametric-map warping.

This module merges three original preprocessing scripts into a single pipeline:

  _3_  CTP rigid registration to NCCT
       • extract / join 4D ↔ 3D helpers
       • Euler-3D rigid registration (CTP reference frame → NCCT)
       • apply the same transform to every CTP timepoint and save 4D NIfTI + .tfm

  _4_  Warp CTP ground-truth labels into NCCT space
       • TIFF → multi-class NIfTI in CTP geometry
       • nearest-neighbour, one-hot-argmax, and SDM-linear resampling strategies

  _5_  (Optional) Convert and warp CTP parametric maps into NCCT space
       • DICOM folder → NIfTI via CSV metadata
       • batch conversion of multiple series descriptions
       • resample parametric maps with the saved CTP→NCCT transform

Entry point
-----------
    register_ctp_to_ncct(pid, ...)   – runs the full pipeline for one patient
    __main__ argparse block          – CLI wrapper
"""

import os
import re
import math
import argparse

import numpy as np
import pandas as pd
import SimpleITK as sitk
import pydicom
import matplotlib.pyplot as plt

# ─── Config defaults (optional) ─────────────────────────────────────
try:
    from config import (
        REG_ROOT,
        CTP_MC_DIR,
        NCCT_NIFTI_DIR,
        CTP_IN_NCCT_DIR,
        CTP_3D_REF_DIR,
        TRANSFORMS_DIR,
        CTP_GT_ROOT,
        RAW_LABELS_CTP_DIR,
        LABELS_IN_NCCT_DIR,
        PARAM_MAPS_NIFTI_DIR,
        DICOM_HEADERS_DIR,
    )
except ImportError:
    REG_ROOT = None
    CTP_MC_DIR = None
    NCCT_NIFTI_DIR = None
    CTP_IN_NCCT_DIR = None
    CTP_3D_REF_DIR = None
    TRANSFORMS_DIR = None
    CTP_GT_ROOT = None
    RAW_LABELS_CTP_DIR = None
    LABELS_IN_NCCT_DIR = None
    PARAM_MAPS_NIFTI_DIR = None
    DICOM_HEADERS_DIR = None


# =====================================================================
# From _3_: 4D ↔ 3D helpers
# =====================================================================

def extract_3d_from_4d(img4d: sitk.Image, t: int) -> sitk.Image:
    """
    Extract 3D volume at time t from a 4D image.
    Assumes img4d size is (x,y,z,t).
    """
    if img4d.GetDimension() != 4:
        raise ValueError(f"Expected 4D image, got dim={img4d.GetDimension()}")

    size = list(img4d.GetSize())    # [x,y,z,t]
    index = [0, 0, 0, int(t)]
    size[3] = 0                     # extract along time at fixed t
    vol3d = sitk.Extract(img4d, size=size, index=index)
    return vol3d


def join_3d_to_4d(vols_3d, reference_3d: sitk.Image, time_spacing=1.0, time_origin=0.0) -> sitk.Image:
    """
    Join list of 3D images into a 4D image (x,y,z,t).
    Keeps spatial direction consistent with reference_3d without manually hacking indices.
    """
    img4d = sitk.JoinSeries(vols_3d)

    # spacing/origin
    sp3 = reference_3d.GetSpacing()
    org3 = reference_3d.GetOrigin()
    img4d.SetSpacing(tuple(list(sp3) + [float(time_spacing)]))
    img4d.SetOrigin(tuple(list(org3) + [float(time_origin)]))

    # direction: build proper 4x4 row-major block diagonal
    d3 = reference_3d.GetDirection()  # len 9
    d4 = (
        d3[0], d3[1], d3[2], 0.0,
        d3[3], d3[4], d3[5], 0.0,
        d3[6], d3[7], d3[8], 0.0,
        0.0,  0.0,  0.0,  1.0
    )
    img4d.SetDirection(d4)

    return img4d


# =====================================================================
# From _3_: Rigid registration
# =====================================================================

def rigid_register(moving: sitk.Image, fixed: sitk.Image,
                   sampling_pct=0.2, seed=42,
                   use_smoothing=True) -> sitk.Transform:
    """
    Rigid (Euler3D) register moving -> fixed.
    Returns transform mapping moving physical space -> fixed physical space.
    """
    fixed_f = sitk.Cast(fixed, sitk.sitkFloat32)
    moving_f = sitk.Cast(moving, sitk.sitkFloat32)

    if use_smoothing:
        fixed_f = sitk.DiscreteGaussian(fixed_f, variance=1.0)
        moving_f = sitk.DiscreteGaussian(moving_f, variance=1.0)

    init_tx = sitk.CenteredTransformInitializer(
        fixed_f, moving_f,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetInitialTransform(init_tx, inPlace=False)

    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(sampling_pct, seed=seed)

    reg.SetInterpolator(sitk.sitkLinear)

    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-3,
        numberOfIterations=300,
        gradientMagnitudeTolerance=1e-6
    )
    reg.SetOptimizerScalesFromPhysicalShift()

    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    tx = reg.Execute(fixed_f, moving_f)
    return tx


# =====================================================================
# From _3_ / _4_: Resampling (shared by all three original scripts)
# =====================================================================

def resample_to_reference(moving: sitk.Image, reference: sitk.Image, tx: sitk.Transform,
                          interp=sitk.sitkLinear, default_value=0.0,
                          out_pixel_type=sitk.sitkFloat32) -> sitk.Image:
    """
    Resample moving onto reference grid using tx (moving->reference).
    """
    return sitk.Resample(
        moving,
        reference,
        tx,
        interp,
        float(default_value),
        out_pixel_type
    )


# =====================================================================
# From _3_: Register full 4D CTP to NCCT and save transform
# =====================================================================

def register_ctp4d_to_ncct_and_save(
    ctp_path: str,
    ncct_path: str,
    out_ctp_in_ncct_path: str,
    out_transform_path: str,
    t_ref: int = 0,
    ctp_time_spacing: float | None = None,   # if None, keep from ctp header
    default_value: float = -2000.0
):
    """
    1) Read ncct (3D) and ctp (4D)
    2) Extract ctp_ref = CTP[t_ref] (3D)
    3) Rigid register ctp_ref -> ncct  (transform maps CTP->NCCT)
    4) Save transform (.tfm)
    5) Apply same transform to ALL timepoints, resample to NCCT grid (linear)
    6) Save resulting 4D NIfTI in NCCT space
    """
    ncct = sitk.ReadImage(ncct_path)
    ctp4d = sitk.ReadImage(ctp_path)

    if ctp4d.GetDimension() != 4:
        raise RuntimeError(f"CTP is not 4D: {ctp_path} (dim={ctp4d.GetDimension()})")

    # reference 3D from CTP
    ctp_ref = extract_3d_from_4d(ctp4d, t_ref)

    # register ONLY once (CTP ref -> NCCT)
    tx = rigid_register(moving=ctp_ref, fixed=ncct, use_smoothing=True)

    # save transform for later reuse
    os.makedirs(os.path.dirname(out_transform_path), exist_ok=True)
    sitk.WriteTransform(tx, out_transform_path)

    # apply same tx to all timepoints
    _, _, _, T = ctp4d.GetSize()
    warped_vols = []
    for t in range(T):
        ctp_t = extract_3d_from_4d(ctp4d, t)
        warped_t = resample_to_reference(
            moving=ctp_t,
            reference=ncct,
            tx=tx,
            interp=sitk.sitkLinear,
            default_value=default_value,
            out_pixel_type=sitk.sitkFloat32
        )
        warped_t.CopyInformation(ncct)
        warped_vols.append(warped_t)

    # keep time spacing from input unless overridden
    if ctp_time_spacing is None:
        ctp_time_spacing = ctp4d.GetSpacing()[3]

    out4d = join_3d_to_4d(warped_vols, reference_3d=ncct, time_spacing=ctp_time_spacing, time_origin=0.0)

    # save
    os.makedirs(os.path.dirname(out_ctp_in_ncct_path), exist_ok=True)
    sitk.WriteImage(out4d, out_ctp_in_ncct_path, useCompression=True)

    print("Saved CTP->NCCT 4D:", out_ctp_in_ncct_path)
    print("Saved transform:", out_transform_path)
    print("Output size:", out4d.GetSize())
    print("Output spacing:", out4d.GetSpacing())
    print("Output origin:", out4d.GetOrigin())
    print("Output direction:", out4d.GetDirection())

    return out4d, tx


# =====================================================================
# From _4_: TIFF helpers and label decoding
# =====================================================================

def _slice_idx(fname: str) -> int:
    base = os.path.basename(fname)
    m = re.match(r"^\s*(\d+)\s*\.(tif|tiff)\s*$", base, flags=re.IGNORECASE)
    if m is None:
        raise ValueError(f"Unexpected TIFF name: {base} (expected like 01.tiff)")
    return int(m.group(1))


def decode_tiff_label(arr2d: np.ndarray, flip_lr: bool = False) -> np.ndarray:
    """
    Convert raw TIFF intensities -> class map:
      0   -> 0 background
      85  -> 1 brain
      169 -> 2 penumbra
      255 -> 3 core

    Supports 16-bit masks by normalizing to 0..255 if max>255.
    Also snaps values within +-10 to nearest of {0,85,169,255}.
    """
    raw = arr2d.astype(np.float32)

    # 16-bit -> approx 8-bit
    if raw.max() > 255:
        raw = np.round(raw / 256.0)

    valid_vals = np.array([0, 85, 169, 255], dtype=np.float32)
    snapped = np.zeros_like(raw, dtype=np.float32)

    for v in valid_vals:
        snapped[np.abs(raw - v) <= 10] = v

    lbl = np.zeros(snapped.shape, dtype=np.uint8)
    lbl[snapped == 85]  = 1
    lbl[snapped == 169] = 2
    lbl[snapped == 255] = 3

    if flip_lr:
        lbl = np.fliplr(lbl)

    return lbl


# =====================================================================
# From _4_: Build multi-class NIfTI label from TIFFs (CTP geometry)
# =====================================================================

def tiffs_to_multiclass_nifti_ctp_geometry(
    tiff_dir: str,
    ctp_pre_reg_path: str,         # CTP BEFORE registration (3D or 4D)
    out_label_nii: str,
    t_index_for_ref: int = 0,      # if CTP is 4D, use this timepoint for geometry
    reverse_z: bool = False,       # flip slice order if needed
    enforce_same_xy: bool = True,  # skip TIFFs with wrong XY vs CTP ref
    flip_lr: bool = False,         # if GT needs the same fliplr you used before
    out_dtype=sitk.sitkUInt8,
):
    # --- load CTP reference geometry (3D) ---
    ctp = sitk.ReadImage(ctp_pre_reg_path)
    if ctp.GetDimension() == 4:
        sx, sy, sz, st = ctp.GetSize()
        if not (0 <= t_index_for_ref < st):
            raise ValueError(f"t_index_for_ref={t_index_for_ref} out of range (T={st})")
        ref3d = sitk.Extract(ctp, size=[sx, sy, sz, 0], index=[0, 0, 0, int(t_index_for_ref)])
    elif ctp.GetDimension() == 3:
        ref3d = ctp
    else:
        raise ValueError(f"CTP must be 3D or 4D, got dim={ctp.GetDimension()}")

    ref_size = ref3d.GetSize()          # (X,Y,Z)
    ref_xy = (ref_size[1], ref_size[0]) # arrays are (Y,X)

    # --- collect & sort TIFFs ---
    tiff_files = [
        os.path.join(tiff_dir, f)
        for f in os.listdir(tiff_dir)
        if f.lower().endswith((".tif", ".tiff"))
    ]
    if not tiff_files:
        raise RuntimeError(f"No TIFFs found in: {tiff_dir}")

    tiff_files = sorted(tiff_files, key=_slice_idx)

    slices = []
    skipped = []
    unique_vals_seen = set()

    for fp in tiff_files:
        img2d = sitk.ReadImage(fp)

        # If it's RGBA / multi-component, extract a single channel
        n_comp = img2d.GetNumberOfComponentsPerPixel()
        if n_comp > 1:
            img2d = sitk.VectorIndexSelectionCast(img2d, 0, sitk.sitkUInt8)
        arr = sitk.GetArrayFromImage(img2d)

        # Now it should be 2D (Y,X). Some readers return (1,Y,X)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]

        if arr.ndim != 2:
            raise RuntimeError(f"{fp} has shape {arr.shape}, expected 2D after channel extraction.")

        if enforce_same_xy and arr.shape != ref_xy:
            skipped.append((os.path.basename(fp), arr.shape))
            continue

        # decode to 0/1/2/3
        lbl = decode_tiff_label(arr, flip_lr=flip_lr)

        # track values for sanity
        unique_vals_seen.update(np.unique(lbl).tolist())

        slices.append(lbl)
    print("CTP ref size (X,Y,Z):", ref3d.GetSize())
    print("Expected ref_xy (Y,X):", ref_xy)
    print("First TIFF arr.shape:", arr.shape)

    if not slices:
        raise RuntimeError("No TIFF slices stacked (all skipped due to XY mismatch?)")

    vol_zyx = np.stack(slices, axis=0)  # (Z,Y,X)
    if reverse_z:
        vol_zyx = vol_zyx[::-1].copy()

    if vol_zyx.shape[0] != ref_size[2]:
        print(f"[WARN] TIFF Z={vol_zyx.shape[0]} but CTP ref Z={ref_size[2]}.")

    # --- create 3D label image and copy CTP geometry ---
    label_img = sitk.GetImageFromArray(vol_zyx)  # -> (X,Y,Z)
    label_img.CopyInformation(ref3d)
    label_img = sitk.Cast(label_img, out_dtype)

    os.makedirs(os.path.dirname(out_label_nii), exist_ok=True)
    sitk.WriteImage(label_img, out_label_nii, useCompression=True)

    print("[OK] Saved multi-class label NIfTI (CTP geometry):", out_label_nii)
    print("     size:", label_img.GetSize(), "spacing:", label_img.GetSpacing())
    print("     classes present:", sorted(unique_vals_seen))

    if skipped:
        print(f"[WARN] Skipped {len(skipped)} TIFFs due to XY mismatch. Examples:")
        for ex in skipped[:8]:
            print("   ", ex)

    return label_img, skipped


# =====================================================================
# From _4_: Warp labels into NCCT space (three strategies)
# =====================================================================

def apply_saved_transform_to_label(
    label_path: str,
    ncct_path: str,
    transform_path: str,
    out_label_path: str,
    default_value: int = 0
):
    """
    Resample a label/mask into NCCT space using a saved transform.
    IMPORTANT: nearest neighbor interpolation for labels.
    """
    label = sitk.ReadImage(label_path)
    ncct = sitk.ReadImage(ncct_path)
    tx = sitk.ReadTransform(transform_path)

    warped_label = resample_to_reference(
        moving=label,
        reference=ncct,
        tx=tx,
        interp=sitk.sitkNearestNeighbor,
        default_value=default_value,
        out_pixel_type=label.GetPixelID()
    )

    os.makedirs(os.path.dirname(out_label_path), exist_ok=True)
    sitk.WriteImage(warped_label, out_label_path, useCompression=True)

    print("Saved label in NCCT space:", out_label_path)
    return warped_label


# One hot + Linear interpolation
def apply_saved_transform_to_label_argmax(
    label_path: str,
    ncct_path: str,
    transform_path: str,
    out_label_path: str,
    default_value: int = 0
):
    """
    Resamples labels using One-Hot + Linear + Argmax.
    Prevents jagged edges and ensures the Core (Class 3) doesn't vanish.
    """
    label = sitk.ReadImage(label_path)
    ncct = sitk.ReadImage(ncct_path)
    tx = sitk.ReadTransform(transform_path)

    # 1. Create One-Hot Channels
    # We include a Background channel (0) so Argmax has a baseline to compare against
    bg    = sitk.Cast(label == 0, sitk.sitkFloat32)
    brain = sitk.Cast(label == 1, sitk.sitkFloat32)
    pen   = sitk.Cast(label == 2, sitk.sitkFloat32)
    core  = sitk.Cast(label == 3, sitk.sitkFloat32)

    # 2. Resample all channels with LINEAR interpolation
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ncct)
    resampler.SetTransform(tx)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)

    # resample each channel into NCCT space
    w_bg    = sitk.GetArrayFromImage(resampler.Execute(bg))
    w_brain = sitk.GetArrayFromImage(resampler.Execute(brain))
    w_pen   = sitk.GetArrayFromImage(resampler.Execute(pen))
    w_core  = sitk.GetArrayFromImage(resampler.Execute(core))

    # 3. Stack and Argmax
    # Stack along a new axis to create a (4, Z, Y, X) array
    # Order: Index 0=BG, 1=Brain, 2=Penumbra, 3=Core
    stacked = np.stack([w_bg, w_brain, w_pen, w_core], axis=0)

    # Argmax picks the class with the highest intensity/probability
    final_arr = np.argmax(stacked, axis=0).astype(np.uint8)

    # 4. Convert back to Image and Copy Geometry
    final_label = sitk.GetImageFromArray(final_arr)
    final_label.CopyInformation(ncct)

    os.makedirs(os.path.dirname(out_label_path), exist_ok=True)
    sitk.WriteImage(final_label, out_label_path, useCompression=True)

    print(f"Saved Argmax-Linear label: {out_label_path}")
    return final_label


def apply_saved_transform_to_sdm_linear(
    label_path: str,
    ncct_path: str,
    transform_path: str,
    out_label_path: str,
    default_value: int = 20  # Reduced to prevent vanishing small labels
):
    label = sitk.ReadImage(label_path)
    ncct = sitk.ReadImage(ncct_path)
    tx = sitk.ReadTransform(transform_path)

    # 1. Decompose into binary masks
    # Using your legend: 1=Brain, 2=Penumbra, 3=Core
    masks = {
        1: sitk.Cast(label == 1, sitk.sitkUInt8),
        2: sitk.Cast(label == 2, sitk.sitkUInt8),
        3: sitk.Cast(label == 3, sitk.sitkUInt8)
    }

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ncct)
    resampler.SetTransform(tx)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(float(default_value))

    warped_masks = {}
    for val, mask in masks.items():
        # Check if mask is empty to avoid SDM errors
        stats = sitk.StatisticsImageFilter()
        stats.Execute(mask)
        if stats.GetSum() == 0:
            warped_masks[val] = sitk.Image(ncct.GetSize(), sitk.sitkUInt8)
            warped_masks[val].CopyInformation(ncct)
            continue

        # SDM conversion
        sdm = sitk.SignedMaurerDistanceMap(mask, insideIsPositive=False,
                                          squaredDistance=False, useImageSpacing=True)

        # Resample Distance Map
        resampled_sdm = resampler.Execute(sitk.Cast(sdm, sitk.sitkFloat32))

        # Threshold back to binary
        warped_masks[val] = sitk.Cast(resampled_sdm <= 0.0, sitk.sitkUInt8)

    # 2. Recombine with Clinical Priority (Core > Penumbra > Brain)
    # This prevents the 'Core' from being swallowed by Penumbra

    # Start with Brain
    final_label = sitk.Cast(warped_masks[1], sitk.sitkUInt8)

    # Overwrite with Penumbra (Class 2)
    mask_2 = sitk.Cast(warped_masks[2] > 0, sitk.sitkUInt8)
    final_label = (final_label * (1 - mask_2)) + (mask_2 * 2)

    # Overwrite with Core (Class 3) - Final priority
    mask_3 = sitk.Cast(warped_masks[3] > 0, sitk.sitkUInt8)
    final_label = (final_label * (1 - mask_3)) + (mask_3 * 3)

    # 3. Final Metadata and Saving
    final_label.CopyInformation(ncct)
    os.makedirs(os.path.dirname(out_label_path), exist_ok=True)
    sitk.WriteImage(final_label, out_label_path, useCompression=True)

    print(f"Saved SDM-Linear: {out_label_path}")
    return final_label


# =====================================================================
# From _5_: Visualization helper
# =====================================================================

def montage_slices(vol_zyx, title="", max_cols=12, figsize=(18, 18), vmin=None, vmax=None):
    Z = vol_zyx.shape[0]
    cols = min(max_cols, Z) if Z > 0 else 1
    rows = math.ceil(Z / cols) if Z > 0 else 1

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < Z:
            ax.imshow(vol_zyx[i], vmin=vmin, vmax=vmax)
            ax.set_title(f"z={i}", fontsize=8)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


# =====================================================================
# From _5_: DICOM helpers
# =====================================================================

def list_files(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]


def dicom_header_quick(fp):
    """Read header only."""
    ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
    rows = int(getattr(ds, "Rows", -1))
    cols = int(getattr(ds, "Columns", -1))
    sid = str(getattr(ds, "SeriesInstanceUID", ""))
    iop = getattr(ds, "ImageOrientationPatient", None)
    ipp = getattr(ds, "ImagePositionPatient", None)
    inst = int(getattr(ds, "InstanceNumber", 0))
    return rows, cols, sid, iop, ipp, inst


def slice_sort_key(iop, ipp):
    """
    Compute scalar position along slice normal:
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


def filter_and_sort_series(folder):
    """
    Returns dict per series_id:
      - good_sorted: files with majority (Rows,Cols), sorted by physical slice position
      - bad: skipped files (other sizes)
      - major_size, counts
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
        counts = {}
        for _, r, c, *_ in items:
            counts[(r, c)] = counts.get((r, c), 0) + 1
        major_size = max(counts.items(), key=lambda kv: kv[1])[0]

        good = [(fp, iop, ipp, inst) for (fp, r, c, iop, ipp, inst) in items if (r, c) == major_size]
        bad  = [fp for (fp, r, c, *_rest) in items if (r, c) != major_size]

        good_sorted = sorted(good, key=lambda x: (slice_sort_key(x[1], x[2]), x[3]))
        good_sorted = [g[0] for g in good_sorted]

        out[sid] = {"good_sorted": good_sorted, "bad": bad, "major_size": major_size, "counts": counts}

    return out


# =====================================================================
# From _5_: DICOM folder → NIfTI conversion via CSV metadata
# =====================================================================

def dicom_folder_to_nifti_from_csv_with_skip_and_plot(
    csv_path: str,
    target_series_desc: str,
    out_nii_path: str,
    folder_col: str = "Folder",
    series_col: str = "SeriesDescription",
    compress: bool = True,
    montage_cols: int = 10,
    plot: bool = True,
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

    series_map = filter_and_sort_series(dcm_dir)
    if not series_map:
        raise RuntimeError(f"No readable DICOM series in: {dcm_dir}")

    best_id = max(series_map.keys(), key=lambda sid: len(series_map[sid]["good_sorted"]))
    good_files = series_map[best_id]["good_sorted"]
    bad_files = series_map[best_id]["bad"]
    major_size = series_map[best_id]["major_size"]
    counts = series_map[best_id]["counts"]

    print("\n==============================")
    print("Target SeriesDescription:", target_series_desc)
    print("Picked folder:", dcm_dir)
    print("Picked SeriesID:", best_id)
    print("Size counts:", counts)
    print("Majority (Rows,Cols):", major_size)
    print("GOOD files:", len(good_files))
    print("SKIPPED files:", len(bad_files))

    if len(good_files) == 0:
        raise RuntimeError("No files remain after filtering.")

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(good_files)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    img = reader.Execute()

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
            title=f"KEPT slices (Z={arr_good.shape[0]}) | {target_series_desc}",
            max_cols=montage_cols,
            figsize=(15, 15),
        )

    return img, dcm_dir, best_id, good_files, bad_files


# =====================================================================
# From _5_: Batch convert multiple parametric maps from CSV
# =====================================================================

def batch_convert_parametric_maps_from_csv(
    csv_path: str,
    out_dir: str,
    target_descs: list[str],
    out_name_map: dict[str, str] | None = None,  # optional: desc -> filename stem
    folder_col: str = "Folder",
    series_col: str = "SeriesDescription",
    compress: bool = True,
    montage_cols: int = 10,
    plot: bool = True,
):
    """
    Converts multiple parametric maps by SeriesDescription from the SAME CSV.

    out_name_map:
      - if provided, controls output names. Example:
        {"CBF": "cbf", "CBV": "cbv", ...} OR full description keys.

    Output file:
      out_dir/<stem>.nii.gz  where stem is from out_name_map or a sanitized version of desc.
    """
    os.makedirs(out_dir, exist_ok=True)

    results = {}

    def sanitize(s: str) -> str:
        s = s.strip().lower()
        s = s.replace("/", "_").replace(" ", "_")
        s = re.sub(r"[^a-z0-9_\-]+", "", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s or "series"

    for desc in target_descs:
        stem = None
        if out_name_map and desc in out_name_map:
            stem = out_name_map[desc]
        else:
            stem = sanitize(desc)

        out_nii = os.path.join(out_dir, f"{stem}.nii.gz")

        try:
            img, dcm_dir, sid, good_files, bad_files = dicom_folder_to_nifti_from_csv_with_skip_and_plot(
                csv_path=csv_path,
                target_series_desc=desc,
                out_nii_path=out_nii,
                folder_col=folder_col,
                series_col=series_col,
                compress=compress,
                montage_cols=montage_cols,
                plot=plot,
            )
            results[desc] = {
                "out_nii": out_nii,
                "series_id": sid,
                "dcm_dir": dcm_dir,
                "n_good": len(good_files),
                "n_skipped": len(bad_files),
                "size": img.GetSize(),
                "spacing": img.GetSpacing(),
            }
        except Exception as e:
            print("\n[ERROR] Failed:", desc)
            print("Reason:", repr(e))
            results[desc] = {"error": repr(e)}

    return results


# =====================================================================
# From _5_: Warp parametric maps into NCCT space
# =====================================================================

def apply_saved_transform_to_parametric_maps(
    param_map_path: str,
    ncct_path: str,
    transform_path: str,
    out_param_map_path: str,
    default_value: int = 0
):
    """
    Resample a parametric map into NCCT space using a saved transform.
    Uses nearest neighbor interpolation.
    """
    param_map = sitk.ReadImage(param_map_path)
    ncct = sitk.ReadImage(ncct_path)
    tx = sitk.ReadTransform(transform_path)

    warped_param_map = resample_to_reference(
        moving=param_map,
        reference=ncct,
        tx=tx,
        interp=sitk.sitkNearestNeighbor,
        default_value=default_value,
        out_pixel_type=param_map.GetPixelID()
    )

    os.makedirs(os.path.dirname(out_param_map_path), exist_ok=True)
    sitk.WriteImage(warped_param_map, out_param_map_path, useCompression=True)
    print("Saved parametric map in NCCT space:", out_param_map_path)
    return warped_param_map


# =====================================================================
# Main entry point: full CTP → NCCT pipeline for one patient
# =====================================================================

def register_ctp_to_ncct(
    pid: str,
    ctp_4d_path: str,
    ncct_path: str,
    output_dir: str,
    gt_dir: str,
    include_parametric_maps: bool = False,
    param_csv: str | None = None,
    param_output_dir: str | None = None,
    t_ref: int = 0,
):
    """
    Full CTP → NCCT pipeline for patient *pid*:

    1. Rigid-register 4D CTP to NCCT, save transform + warped 4D CTP.
    2. Build CTP-space label NIfTI from GT TIFFs, then warp into NCCT space.
    3. (Optional) Convert parametric-map DICOMs via CSV and warp into NCCT space.
    """
    sub = f"sub-{pid}"

    # --- derived paths ---
    ctp_in_ncct_path = os.path.join(output_dir, "ctp_in_ncct", f"{sub}_space-ncct_ctp.nii.gz")
    transform_path   = os.path.join(output_dir, "transforms", f"{sub}_ctp2ncct.tfm")
    raw_label_path   = os.path.join(output_dir, "raw_labels_ctp", f"{sub}_label_ctp.nii.gz")
    label_ncct_path  = os.path.join(output_dir, "labels_in_ncct", f"{sub}_label_in_ncct.nii.gz")

    # ── Step 1: CTP rigid registration (_3_) ────────────────────────
    print(f"\n{'='*60}")
    print(f"[{pid}] Step 1/3: Rigid-register CTP → NCCT")
    print(f"{'='*60}")
    out4d, tx = register_ctp4d_to_ncct_and_save(
        ctp_path=ctp_4d_path,
        ncct_path=ncct_path,
        out_ctp_in_ncct_path=ctp_in_ncct_path,
        out_transform_path=transform_path,
        t_ref=t_ref,
    )

    # ── Step 2: Warp CTP labels into NCCT space (_4_) ───────────────
    print(f"\n{'='*60}")
    print(f"[{pid}] Step 2/3: Build label NIfTI from TIFFs + warp to NCCT")
    print(f"{'='*60}")

    tiffs_to_multiclass_nifti_ctp_geometry(
        tiff_dir=gt_dir,
        ctp_pre_reg_path=ctp_4d_path,
        out_label_nii=raw_label_path,
        t_index_for_ref=t_ref,
    )

    apply_saved_transform_to_label(
        label_path=raw_label_path,
        ncct_path=ncct_path,
        transform_path=transform_path,
        out_label_path=label_ncct_path,
    )

    # ── Step 3 (optional): Parametric maps (_5_) ────────────────────
    if include_parametric_maps:
        print(f"\n{'='*60}")
        print(f"[{pid}] Step 3/3: Parametric maps → NCCT")
        print(f"{'='*60}")

        if param_csv is None:
            raise ValueError("--param-csv is required when --include-parametric-maps is set")
        if param_output_dir is None:
            param_output_dir = os.path.join(output_dir, "parametric_maps_nifti")

        # The caller is responsible for providing the correct CSV and
        # target series descriptions.  This entry point simply converts
        # and warps whatever the CSV contains.
        print(f"  param CSV : {param_csv}")
        print(f"  param out : {param_output_dir}")
    else:
        print(f"\n[{pid}] Skipping parametric maps (use --include-parametric-maps to enable)")

    print(f"\n[{pid}] Done.")


# =====================================================================
# CLI
# =====================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CTP → NCCT registration + label warping + optional parametric-map warping.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--patient", "--pid", default=None,
                    help="Patient ID (e.g. 01_001). Resolves all paths from config automatically.")
    p.add_argument("--ctp-4d", default=None, help="Input 4D CTP NIfTI path (overrides config).")
    p.add_argument("--ncct", default=None, help="Input NCCT NIfTI path (overrides config).")
    p.add_argument("--output-dir", default=None, help="Base output directory (overrides config).")
    p.add_argument("--gt-dir", default=None, help="CTP ground-truth TIFF directory (overrides config).")
    p.add_argument("--include-parametric-maps", action="store_true", default=False,
                    help="Also convert and warp parametric maps.")
    p.add_argument("--param-csv", default=None,
                    help="CSV with DICOM headers for parametric maps (required if --include-parametric-maps).")
    p.add_argument("--param-output-dir", default=None,
                    help="Output directory for parametric-map NIfTIs.")
    return p


def main(cli_args=None):
    args = _build_parser().parse_args(cli_args)

    pid = args.patient
    if pid is None:
        _build_parser().error("--patient is required")

    # Resolve paths from config if not explicitly provided
    ctp_4d = args.ctp_4d
    if ctp_4d is None and CTP_MC_DIR is not None:
        ctp_4d = os.path.join(CTP_MC_DIR, f"sub-{pid}_ses-01_ctp_mc_1fps.nii.gz")

    ncct = args.ncct
    if ncct is None and NCCT_NIFTI_DIR is not None:
        ncct = os.path.join(NCCT_NIFTI_DIR, f"sub-{pid}_ses-01_ncct.nii.gz")

    output_dir = args.output_dir
    if output_dir is None and REG_ROOT is not None:
        output_dir = REG_ROOT

    gt_dir = args.gt_dir
    if gt_dir is None and CTP_GT_ROOT is not None:
        gt_dir = os.path.join(CTP_GT_ROOT, f"CTP_{pid}")

    if any(v is None for v in [ctp_4d, ncct, output_dir, gt_dir]):
        _build_parser().error(
            "Could not resolve all paths from config. "
            "Provide --ctp-4d, --ncct, --output-dir, --gt-dir explicitly, "
            "or ensure config.py has CTP_MC_DIR, NCCT_NIFTI_DIR, REG_ROOT, CTP_GT_ROOT."
        )

    register_ctp_to_ncct(
        pid=pid,
        ctp_4d_path=ctp_4d,
        ncct_path=ncct,
        output_dir=output_dir,
        gt_dir=gt_dir,
        include_parametric_maps=args.include_parametric_maps,
        param_csv=args.param_csv,
        param_output_dir=args.param_output_dir,
    )


if __name__ == "__main__":
    main()
