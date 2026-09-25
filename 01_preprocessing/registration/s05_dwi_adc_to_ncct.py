"""
s05_dwi_adc_to_ncct.py
======================
DWI → NCCT registration, DWI lesion mask warping, and ADC → NCCT warping.

This module merges three original preprocessing scripts into a single pipeline:

  _7_  Multi-start rigid + affine DWI-to-NCCT registration
       • NCCT fixed mask (soft-tissue extraction)
       • DWI moving mask (Otsu + morphology)
       • multi-start Euler3D rigid search with rotational perturbations
       • affine refinement seeded from best rigid result
       • resample DWI onto NCCT grid and save transform

  _8_  Warp DWI lesion mask PNG slices to NCCT space
       • optional PNG half-selection for thick-slice DWI
       • stack 2-D PNG masks → 3-D NIfTI in DWI geometry
       • apply saved DWI→NCCT transform (nearest-neighbour) to warp mask

  _9_  Register ADC maps to NCCT using DWI-to-NCCT transform
       • resample ADC volume into NCCT space with the saved affine
       • skull-strip using NCCT brain mask

Entry point
-----------
    register_dwi_to_ncct(pid, ...)   – runs the full pipeline for one patient
    __main__ argparse block          – CLI wrapper

Standalone usage
----------------
    python s05_dwi_adc_to_ncct.py \\
        --pid 01_001 \\
        --dwi /path/to/dwi.nii.gz \\
        --ncct /path/to/ncct.nii.gz \\
        --output-dir /path/to/output
"""

import os
import math
import glob
import argparse

import numpy as np
import SimpleITK as sitk
import itk

# ─── Config defaults (optional) ─────────────────────────────────────
try:
    from config import (
        REG_ROOT,
        CURATED_RAW_NIFTI_DIR,
        DWI_NIFTI_DIR,
        NCCT_NIFTI_DIR,
        TRANSFORMS_DIR,
        DWI_IN_NCCT_DIR,
        DWI_MASKS_DIR,
        DWI_GT_ROOT,
        ADC_RAW_DIR,
        ADC_REG_DIR,
        NCCT_SKULL_STRIPPED_DIR,
        BRAIN_MASKS_DIR,
    )
except ImportError:
    REG_ROOT = None
    CURATED_RAW_NIFTI_DIR = None
    DWI_NIFTI_DIR = None
    NCCT_NIFTI_DIR = None
    TRANSFORMS_DIR = None
    DWI_IN_NCCT_DIR = None
    DWI_MASKS_DIR = None
    DWI_GT_ROOT = None
    ADC_RAW_DIR = None
    ADC_REG_DIR = None
    NCCT_SKULL_STRIPPED_DIR = None
    BRAIN_MASKS_DIR = None

# Aliases used internally by helper functions
DWI_BRAIN_MASKS_DIR = DWI_MASKS_DIR
NCCT_SS_DIR = NCCT_SKULL_STRIPPED_DIR
BRAIN_MASK_DIR = BRAIN_MASKS_DIR
ADC_REGISTERED_DIR = ADC_REG_DIR


# =====================================================================
# From _7_: Helpers
# =====================================================================

def largest_cc(binary_img: sitk.Image) -> sitk.Image:
    cc = sitk.ConnectedComponent(binary_img)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)
    if stats.GetNumberOfLabels() == 0:
        out = sitk.Image(binary_img.GetSize(), sitk.sitkUInt8)
        out.CopyInformation(binary_img)
        return out
    largest = max(stats.GetLabels(), key=lambda L: stats.GetPhysicalSize(L))
    out = sitk.Cast(cc == largest, sitk.sitkUInt8)
    out.CopyInformation(binary_img)
    return out


def make_ncct_fixed_mask(pid, ncct_img: sitk.Image,
                         air_thr: float = -200.0,
                         bone_thr: float = 200.0,
                         closing_radius=(6, 6, 2),
                         erode_radius=(2, 2, 1)) -> sitk.Image:
    f = sitk.Cast(ncct_img, sitk.sitkFloat32)

    # 1) Head mask (remove air)
    head = sitk.Cast(f > air_thr, sitk.sitkUInt8)
    head = sitk.BinaryMorphologicalClosing(head, closing_radius)
    head = sitk.BinaryFillhole(head)
    head = largest_cc(head)

    # 2) Soft-tissue inside head (remove skull/bone)
    soft = sitk.Cast((f > air_thr) & (f < bone_thr), sitk.sitkUInt8)
    soft = sitk.Mask(soft, head)
    soft = sitk.BinaryMorphologicalClosing(soft, (4, 4, 2))
    soft = sitk.BinaryFillhole(soft)
    soft = largest_cc(soft)

    # 3) Slight erosion to stay away from skull boundary
    soft = sitk.BinaryErode(soft, erode_radius)

    soft = sitk.Cast(soft, sitk.sitkUInt8)
    soft.CopyInformation(ncct_img)

    if DWI_BRAIN_MASKS_DIR is not None:
        debug_path = os.path.join(DWI_BRAIN_MASKS_DIR, f"brain_mask_ncct_{pid}.nii.gz")
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        sitk.WriteImage(soft, debug_path, useCompression=True)

    return soft


def make_dwi_moving_mask(dwi_img: sitk.Image) -> sitk.Image:
    """
    Robust mask generation for DWI.
    Fixes issues with noise, ghosting artifacts, and 'swiss cheese' holes.
    """
    # 0. Safety: Handle 4D input just in case (Extract b0 volume)
    if dwi_img.GetDimension() == 4:
        extractor = sitk.ExtractImageFilter()
        size = list(dwi_img.GetSize())
        size[3] = 0
        extractor.SetSize(size)
        extractor.SetIndex([0, 0, 0, 0])
        dwi_img = extractor.Execute(dwi_img)

    f = sitk.Cast(dwi_img, sitk.sitkFloat32)

    # 1. SMOOTHING — DWI is grainy; blur so Otsu sees a solid object
    smoother = sitk.SmoothingRecursiveGaussianImageFilter()
    smoother.SetSigma(2.0)  # 2mm smoothing
    f_smooth = smoother.Execute(f)

    # 2. Otsu Threshold on the SMOOTHED image
    mask = sitk.OtsuThreshold(f_smooth, 0, 1)

    # 3. MORPHOLOGICAL OPENING (Disconnect artifacts)
    mask = sitk.BinaryMorphologicalOpening(mask, (3, 3, 1))

    # 4. Fill Holes (Close ventricles, etc.)
    mask = sitk.BinaryFillhole(mask)

    # 5. Keep Largest Component (The Brain)
    mask = largest_cc(mask)

    # 6. DILATE (Safety Margin)
    mask = sitk.BinaryDilate(mask, (3, 3, 1))

    # 7. Final Cast and Info Copy
    mask = sitk.Cast(mask, sitk.sitkUInt8)
    mask.CopyInformation(dwi_img)

    if DWI_MASKS_DIR is not None:
        debug_path = os.path.join(DWI_MASKS_DIR, "debug_dwi_moving_mask.nii.gz")
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        sitk.WriteImage(mask, debug_path, useCompression=True)

    return mask


def unwrap_transform(tx):
    if isinstance(tx, sitk.CompositeTransform):
        return tx.GetNthTransform(tx.GetNumberOfTransforms() - 1)
    return tx


# =====================================================================
# From _7_: Multi-start rigid registration
# =====================================================================

def multistart_rigid_registration(fixed_f, moving_f,
                                  fixed_mask=None, moving_mask=None,
                                  angle_range_deg=15, angle_step_deg=15,
                                  verbose=True):
    """
    Multi-start rigid registration.

    Tries GEOMETRY + MOMENTS initialisations, plus rotational perturbations
    around each axis, runs a quick coarse registration for each, then
    refines the best one with the full pyramid.

    Parameters
    ----------
    angle_range_deg : float
        Max rotation perturbation in degrees (default ±15°).
    angle_step_deg : float
        Step size for perturbation grid (default 15°).

    Returns
    -------
    best_tx : sitk.Transform
    best_metric : float
    """
    # --- Build candidate initial transforms ---
    candidates = []

    for init_mode_name, init_mode in [
        ("GEOMETRY", sitk.CenteredTransformInitializerFilter.GEOMETRY),
        ("MOMENTS",  sitk.CenteredTransformInitializerFilter.MOMENTS),
    ]:
        try:
            base_tx = sitk.CenteredTransformInitializer(
                fixed_f, moving_f,
                sitk.Euler3DTransform(),
                init_mode,
            )
            candidates.append((init_mode_name, base_tx))

            # Add rotational perturbations around the base
            angles = []
            a = -angle_range_deg
            while a <= angle_range_deg:
                if a != 0:
                    angles.append(math.radians(a))
                a += angle_step_deg

            for ax_name, ax_idx in [("X", 0), ("Y", 1), ("Z", 2)]:
                for ang in angles:
                    perturbed = sitk.Euler3DTransform(base_tx)
                    current = list(perturbed.GetParameters())
                    current[ax_idx] += ang
                    perturbed.SetParameters(current)
                    label = f"{init_mode_name}+rot{ax_name}{math.degrees(ang):+.0f}"
                    candidates.append((label, perturbed))

        except RuntimeError:
            if verbose:
                print(f"    [multistart] {init_mode_name} init failed, skipping")

    if not candidates:
        raise RuntimeError("multistart_rigid: no valid initialisations found")

    # --- Quick coarse registration for each candidate ---
    coarse_shrink = [12, 6]
    coarse_sigma  = [6, 3]

    best_metric = 1e9
    best_tx = None
    best_label = ""

    for label, init_tx in candidates:
        try:
            tx, metric = run_registration(
                fixed_f, moving_f,
                fixed_mask=fixed_mask, moving_mask=moving_mask,
                initial_tx=init_tx, stage="rigid",
                shrink_factors=coarse_shrink,
                smooth_sigmas=coarse_sigma,
                max_iterations=100,
            )
            if verbose:
                print(f"    [multistart] {label:>30s}  metric={metric:.6f}")
            if metric < best_metric:
                best_metric = metric
                best_tx = tx
                best_label = label
        except RuntimeError as e:
            if verbose:
                print(f"    [multistart] {label:>30s}  FAILED: {e}")

    if best_tx is None:
        raise RuntimeError("multistart_rigid: all candidates failed")

    if verbose:
        print(f"    [multistart] BEST: {best_label}  metric={best_metric:.6f}")

    # --- Full-resolution refinement from the best coarse result ---
    best_inner = unwrap_transform(best_tx)
    refined_tx, refined_metric = run_registration(
        fixed_f, moving_f,
        fixed_mask=fixed_mask, moving_mask=moving_mask,
        initial_tx=best_inner, stage="rigid",
    )
    if verbose:
        print(f"    [multistart] Refined metric={refined_metric:.6f}")

    return refined_tx, refined_metric


# =====================================================================
# From _7_: Generic registration driver
# =====================================================================

def run_registration(fixed_f, moving_f, fixed_mask=None, moving_mask=None,
                     initial_tx=None, stage="rigid",
                     shrink_factors=None, smooth_sigmas=None,
                     max_iterations=None):
    """
    Parameters
    ----------
    shrink_factors : list[int] or None
        Multi-resolution shrink factors (coarse→fine). Default depends on stage.
    smooth_sigmas : list[float] or None
        Smoothing sigmas in physical units.  Must match len(shrink_factors).
    max_iterations : int or None
        Override default iteration count.
    """
    reg = sitk.ImageRegistrationMethod()

    # --- metric ---
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.15 if stage == "rigid" else 0.25)

    if fixed_mask is not None:
        reg.SetMetricFixedMask(fixed_mask)
    if moving_mask is not None:
        reg.SetMetricMovingMask(moving_mask)

    reg.SetInterpolator(sitk.sitkLinear)

    # --- pyramid (caller can override) ---
    if shrink_factors is None:
        shrink_factors = [8, 4, 2, 1] if stage == "rigid" else [4, 2, 1]
    if smooth_sigmas is None:
        smooth_sigmas = [4, 2, 1, 0] if stage == "rigid" else [2, 1, 0]
    reg.SetShrinkFactorsPerLevel(shrink_factors)
    reg.SetSmoothingSigmasPerLevel(smooth_sigmas)
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # --- optimizer (stage dependent) ---
    if stage == "rigid":
        reg.SetOptimizerAsGradientDescentLineSearch(
            learningRate=1.0,
            numberOfIterations=max_iterations or 300,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10,
        )
    else:  # affine
        reg.SetOptimizerAsGradientDescentLineSearch(
            learningRate=0.5,
            numberOfIterations=max_iterations or 400,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10,
        )

    reg.SetOptimizerScalesFromPhysicalShift()

    # --- init ---
    if initial_tx is None:
        initial_tx = sitk.CenteredTransformInitializer(
            fixed_f, moving_f,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
    reg.SetInitialTransform(initial_tx, inPlace=False)

    tx = reg.Execute(fixed_f, moving_f)
    return tx, reg.GetMetricValue()


# =====================================================================
# From _8_: PNG half-selection helper
# =====================================================================

def maybe_select_png_half(
    mask_png_dir: str,
    pattern: str = "*.png",
    half: str = "first",
    z_threshold: int = 30,
):
    pngs = sorted(glob.glob(os.path.join(mask_png_dir, pattern)))
    if not pngs:
        raise FileNotFoundError(f"No PNGs found in: {mask_png_dir}")

    Z = len(pngs)
    half = half.lower().strip()
    if half not in ("first", "last"):
        raise ValueError('half must be "first" or "last"')

    if Z <= int(z_threshold):
        return pngs, "none", (0, Z), Z

    mid = Z // 2
    if half == "first":
        sel = pngs[:mid]
        z0, z1 = 0, mid
    else:
        sel = pngs[mid:]
        z0, z1 = mid, Z

    return sel, half, (z0, z1), Z


# =====================================================================
# From _8_: PNG slice masks → 3-D NIfTI in DWI geometry
# =====================================================================

def png_masks_to_nifti_in_dwi_space(
    mask_png_dir: str,
    dwi_path: str,
    out_mask_nii_path: str,
    pattern: str = "*.png",
    threshold: int = 0,
    invert: bool = False,
    half: str = "first",
    z_threshold: int = 30,
):
    """
    Reads a folder of 2D PNG mask slices and stacks into a 3D mask.
    Output mask gets SAME spacing/origin/direction as the DWI NIfTI.

    If number of PNG slices > z_threshold, uses first/last half
    (toggle via *half*); otherwise uses all slices.
    """
    dwi = sitk.ReadImage(dwi_path)
    dwi_arr = sitk.GetArrayFromImage(dwi)  # (Z,Y,X)

    # --- choose png list based on threshold/half toggle ---
    pngs, used_half, zrange, Zpng = maybe_select_png_half(
        mask_png_dir=mask_png_dir,
        pattern=pattern,
        half=half,
        z_threshold=z_threshold,
    )

    if used_half != "none":
        print(f"[MASK HALF] PNG Z={Zpng} > {z_threshold} -> using {used_half} half z={zrange[0]}:{zrange[1]}")
    else:
        print(f"[MASK HALF] PNG Z={Zpng} <= {z_threshold} -> using all slices")

    # Read all PNGs as arrays (Y,X)
    slices = []
    for p in pngs:
        img2d = sitk.ReadImage(p)
        arr2d = sitk.GetArrayFromImage(img2d)

        # If PNG loads as (1,Y,X) or (Y,X,3), handle gently
        if arr2d.ndim == 3:
            arr2d = arr2d[..., 0]
        if arr2d.ndim == 2:
            pass
        elif arr2d.ndim == 1:
            raise RuntimeError(f"Unexpected PNG array shape for {p}: {arr2d.shape}")
        else:
            if arr2d.shape[0] == 1:
                arr2d = arr2d[0]
            else:
                raise RuntimeError(f"Unexpected PNG array shape for {p}: {arr2d.shape}")

        # binarize
        bin2d = (arr2d > threshold).astype(np.uint8)
        if invert:
            bin2d = (1 - bin2d).astype(np.uint8)

        slices.append(bin2d)

    mask_arr = np.stack(slices, axis=0)  # (Z,Y,X)

    # Sanity checks vs DWI
    if mask_arr.shape[1:] != dwi_arr.shape[1:]:
        raise ValueError(
            f"Mask XY shape {mask_arr.shape[1:]} != DWI XY shape {dwi_arr.shape[1:]}. "
            "Your PNGs are not in the same in-plane resolution as DWI."
        )

    if mask_arr.shape[0] != dwi_arr.shape[0]:
        raise ValueError(
            f"Mask Z={mask_arr.shape[0]} != DWI Z={dwi_arr.shape[0]}. "
            "If you cropped DWI to half, you must use the same half for PNG masks. "
            f"(Current: used_half={used_half}, zrange={zrange})"
        )

    mask_img = sitk.GetImageFromArray(mask_arr)
    mask_img.CopyInformation(dwi)
    mask_img = sitk.Cast(mask_img, sitk.sitkUInt8)

    os.makedirs(os.path.dirname(out_mask_nii_path), exist_ok=True)
    sitk.WriteImage(mask_img, out_mask_nii_path, useCompression=True)

    print("Saved DWI-space mask NIfTI:", out_mask_nii_path)
    print("Mask size:", mask_img.GetSize(), "spacing:", mask_img.GetSpacing())
    return mask_img, used_half, zrange


# =====================================================================
# From _8_: Warp DWI mask → NCCT using saved transform
# =====================================================================

def warp_dwi_mask_to_ncct(
    dwi_mask_nii_path: str,
    ncct_path: str,
    dwi2ncct_transform_path: str,
    out_mask_in_ncct_path: str,
    default_value: int = 0,
):
    """
    Applies the saved DWI→NCCT transform to a DWI binary mask.
    Uses nearest-neighbor interpolation (required for masks).
    """
    mask = sitk.ReadImage(dwi_mask_nii_path)
    ncct = sitk.ReadImage(ncct_path)
    tx = sitk.ReadTransform(dwi2ncct_transform_path)

    # IMPORTANT: NearestNeighbor to preserve 0/1 labels
    mask_in_ncct = sitk.Resample(
        mask,
        ncct,
        tx,
        sitk.sitkNearestNeighbor,
        int(default_value),
        sitk.sitkUInt8,
    )

    # binarize again in case anything weird happened
    mask_in_ncct = sitk.Cast(mask_in_ncct > 0, sitk.sitkUInt8)

    os.makedirs(os.path.dirname(out_mask_in_ncct_path), exist_ok=True)
    sitk.WriteImage(mask_in_ncct, out_mask_in_ncct_path, useCompression=True)

    print("Saved NCCT-space mask NIfTI:", out_mask_in_ncct_path)
    print("Out size:", mask_in_ncct.GetSize(), "spacing:", mask_in_ncct.GetSpacing())
    return mask_in_ncct


# =====================================================================
# From _9_: ADC helpers
# =====================================================================

def normalize_pid(pid: str) -> str:
    """Strip prefixes like 'sub-', 'stroke' to get bare ID like '01_001'."""
    pid = pid.replace("sub-", "").replace("stroke", "")
    return pid


def find_all_patients() -> list[str]:
    """Discover all patient IDs from the ADC raw data directory."""
    if ADC_RAW_DIR is None:
        raise RuntimeError("ADC_RAW_DIR is not configured")
    pids = []
    for d in sorted(os.listdir(ADC_RAW_DIR)):
        if not os.path.isdir(os.path.join(ADC_RAW_DIR, d)):
            continue
        pid = d.replace("sub-stroke", "").replace("_ses-01", "")
        pids.append(pid)
    return pids


# =====================================================================
# From _9_: Register ADC to NCCT using DWI transform + skull strip
# =====================================================================

def register_adc_to_ncct(pid: str,
                         adc_path: str,
                         tx_path: str,
                         ncct_path: str,
                         mask_path: str,
                         output_dir: str,
                         skip_existing: bool = False):
    """
    Register one patient's ADC map to NCCT space using the saved DWI-to-NCCT
    transform, then apply the NCCT brain mask for skull stripping.
    """
    pid = normalize_pid(pid)

    out_registered = os.path.join(output_dir, f"sub-{pid}_adc_in_ncct.nii.gz")
    out_ss = os.path.join(output_dir, f"sub-{pid}_adc_in_ncct_ss.nii.gz")

    if skip_existing and os.path.exists(out_ss):
        print(f"[SKIP] {pid}: already done -> {out_ss}")
        return

    # --- Check inputs ---
    missing = []
    if not os.path.exists(adc_path):
        missing.append(f"ADC: {adc_path}")
    if not os.path.exists(tx_path):
        missing.append(f"Transform: {tx_path}")
    if not os.path.exists(ncct_path):
        missing.append(f"NCCT: {ncct_path}")
    if not os.path.exists(mask_path):
        missing.append(f"Brain mask: {mask_path}")

    if missing:
        print(f"[SKIP] {pid}: missing files:")
        for m in missing:
            print(f"       - {m}")
        return

    # --- Load ---
    adc = sitk.ReadImage(adc_path)
    ncct = sitk.ReadImage(ncct_path)
    tx = sitk.ReadTransform(tx_path)
    mask = sitk.ReadImage(mask_path)

    # --- Register ADC -> NCCT space ---
    adc_in_ncct = sitk.Resample(
        sitk.Cast(adc, sitk.sitkFloat32),
        ncct,
        tx,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )

    # Save registered (before skull strip)
    os.makedirs(output_dir, exist_ok=True)
    sitk.WriteImage(adc_in_ncct, out_registered, useCompression=True)

    # --- Skull strip with brain mask ---
    mask_bin = sitk.Cast(mask > 0, sitk.sitkFloat32)
    adc_ss = sitk.Multiply(adc_in_ncct, mask_bin)
    adc_ss = sitk.Cast(adc_ss, sitk.sitkFloat32)

    sitk.WriteImage(adc_ss, out_ss, useCompression=True)

    print(f"[OK] {pid}: {out_ss}")


# =====================================================================
# Main entry point: full DWI/mask/ADC → NCCT pipeline for one patient
# =====================================================================

def register_dwi_to_ncct(
    pid: str,
    dwi_path: str,
    ncct_path: str,
    output_dir: str,
    mask_pngs_dir: str | None = None,
    adc_dir: str | None = None,
    skip_existing: bool = False,
    verbose: bool = True,
):
    """
    Full pipeline for one patient:
      1. Multi-start rigid + affine DWI → NCCT registration  (from _7_)
      2. Warp DWI lesion mask PNGs to NCCT space              (from _8_)
      3. Resample ADC into NCCT space using saved transform    (from _9_)

    Parameters
    ----------
    pid : str
        Patient ID (e.g. "01_001").
    dwi_path : str
        Path to DWI NIfTI.
    ncct_path : str
        Path to NCCT NIfTI.
    output_dir : str
        Base output directory; sub-folders are created as needed.
    mask_pngs_dir : str or None
        Directory with DWI lesion mask PNG slices. Skipped if None.
    adc_dir : str or None
        Directory containing ADC raw data (expects adc.nii.gz inside).
        Skipped if None.
    skip_existing : bool
        If True, skip steps whose output files already exist.
    verbose : bool
        Print progress messages.
    """
    pid_clean = normalize_pid(pid)

    # --- Output paths ---
    dwi_out_dir = os.path.join(output_dir, "dwi_registered")
    tx_out_dir = os.path.join(output_dir, "transforms", "dwi")
    mask_out_dir = os.path.join(output_dir, "dwi_masks")
    adc_out_dir = os.path.join(output_dir, "adc_registered")

    out_dwi_in_ncct = os.path.join(dwi_out_dir, f"sub-{pid_clean}_dwi_in_ncct.nii.gz")
    out_tx_path = os.path.join(tx_out_dir, f"sub-{pid_clean}_dwi2ncct_affine.txt")

    # ------------------------------------------------------------------
    # 1) DWI → NCCT registration  (from _7_)
    # ------------------------------------------------------------------
    if skip_existing and os.path.exists(out_dwi_in_ncct) and os.path.exists(out_tx_path):
        if verbose:
            print(f"[SKIP] {pid_clean}: DWI registration already done")
        affine_tx = sitk.ReadTransform(out_tx_path)
    else:
        if verbose:
            print(f"[DWI REG] {pid_clean}: starting multi-start rigid + affine")

        fixed = sitk.ReadImage(ncct_path)
        moving = sitk.ReadImage(dwi_path)

        fixed_f = sitk.Cast(fixed, sitk.sitkFloat32)
        moving_f = sitk.Cast(moving, sitk.sitkFloat32)

        # Build masks
        fixed_mask = make_ncct_fixed_mask(pid_clean, fixed)
        moving_mask = make_dwi_moving_mask(moving)

        # 1a) Multi-start rigid
        rigid_tx, rigid_metric = multistart_rigid_registration(
            fixed_f, moving_f,
            fixed_mask=fixed_mask,
            moving_mask=moving_mask,
            verbose=verbose,
        )
        if verbose:
            print(f"    Rigid metric: {rigid_metric:.6f}")

        # 1b) Affine refinement seeded from best rigid
        rigid_tx_raw = unwrap_transform(rigid_tx)
        affine_init = sitk.AffineTransform(3)
        affine_init.SetCenter(rigid_tx_raw.GetCenter())
        affine_init.SetMatrix(rigid_tx_raw.GetMatrix())
        affine_init.SetTranslation(rigid_tx_raw.GetTranslation())

        affine_tx, affine_metric = run_registration(
            fixed_f, moving_f,
            fixed_mask=fixed_mask,
            initial_tx=affine_init,
            stage="affine",
        )
        if verbose:
            print(f"    Affine metric: {affine_metric:.6f}")

        # Resample DWI → NCCT grid
        dwi_in_ncct = sitk.Resample(
            moving, fixed, affine_tx,
            sitk.sitkLinear, 0.0, moving.GetPixelID(),
        )

        os.makedirs(dwi_out_dir, exist_ok=True)
        os.makedirs(tx_out_dir, exist_ok=True)
        sitk.WriteImage(dwi_in_ncct, out_dwi_in_ncct, useCompression=True)
        sitk.WriteTransform(affine_tx, out_tx_path)

        if verbose:
            print(f"    Saved DWI in NCCT: {out_dwi_in_ncct}")
            print(f"    Saved transform:   {out_tx_path}")

    # ------------------------------------------------------------------
    # 2) Warp DWI lesion mask to NCCT  (from _8_)
    # ------------------------------------------------------------------
    if mask_pngs_dir is not None and os.path.isdir(mask_pngs_dir):
        out_dwi_mask_nii = os.path.join(mask_out_dir, f"sub-{pid_clean}_dwi_mask.nii.gz")
        out_mask_in_ncct = os.path.join(mask_out_dir, f"sub-{pid_clean}_dwi_mask_in_ncct.nii.gz")

        if skip_existing and os.path.exists(out_mask_in_ncct):
            if verbose:
                print(f"[SKIP] {pid_clean}: mask warp already done")
        else:
            if verbose:
                print(f"[MASK WARP] {pid_clean}: PNG masks → NIfTI → NCCT space")

            png_masks_to_nifti_in_dwi_space(
                mask_png_dir=mask_pngs_dir,
                dwi_path=dwi_path,
                out_mask_nii_path=out_dwi_mask_nii,
            )

            warp_dwi_mask_to_ncct(
                dwi_mask_nii_path=out_dwi_mask_nii,
                ncct_path=ncct_path,
                dwi2ncct_transform_path=out_tx_path,
                out_mask_in_ncct_path=out_mask_in_ncct,
            )

            if verbose:
                print(f"    Saved mask in NCCT: {out_mask_in_ncct}")
    elif verbose and mask_pngs_dir is not None:
        print(f"[WARN] {pid_clean}: mask_pngs_dir not found: {mask_pngs_dir}")

    # ------------------------------------------------------------------
    # 3) ADC → NCCT  (from _9_)
    # ------------------------------------------------------------------
    if adc_dir is not None:
        adc_path_candidate = os.path.join(adc_dir, "adc.nii.gz")
        # Also look for skull-stripped NCCT and brain mask from config
        ncct_ss_path = None
        brain_mask_path = None
        if NCCT_SS_DIR is not None:
            ncct_ss_path = os.path.join(NCCT_SS_DIR, f"sub-{pid_clean}_ses-01_ncct_synthstrip.nii.gz")
        if BRAIN_MASK_DIR is not None:
            brain_mask_path = os.path.join(BRAIN_MASK_DIR, f"sub-{pid_clean}_ses-01_ncct_brainmask.nii.gz")

        # Fallback: use the NCCT itself as reference if no skull-stripped version
        ncct_ref = ncct_ss_path if (ncct_ss_path and os.path.exists(ncct_ss_path)) else ncct_path
        mask_ref = brain_mask_path if (brain_mask_path and os.path.exists(str(brain_mask_path))) else None

        if mask_ref is not None:
            if verbose:
                print(f"[ADC REG] {pid_clean}: ADC → NCCT using DWI transform")
            register_adc_to_ncct(
                pid=pid_clean,
                adc_path=adc_path_candidate,
                tx_path=out_tx_path,
                ncct_path=ncct_ref,
                mask_path=mask_ref,
                output_dir=adc_out_dir,
                skip_existing=skip_existing,
            )
        elif verbose:
            print(f"[WARN] {pid_clean}: no brain mask found for ADC skull-strip, skipping ADC")

    if verbose:
        print(f"[DONE] {pid_clean}")


# =====================================================================
# CLI
# =====================================================================

def main(cli_args=None):
    ap = argparse.ArgumentParser(
        description="DWI/ADC → NCCT registration pipeline (steps 7+8+9 merged)",
    )
    ap.add_argument("--patient", "--pid", default=None,
                    help="Patient ID (e.g. 01_001). Resolves all paths from config.")
    ap.add_argument("--dwi", default=None,
                    help="Input DWI NIfTI path (overrides config)")
    ap.add_argument("--ncct", default=None,
                    help="Input NCCT NIfTI path (overrides config)")
    ap.add_argument("--output-dir", default=None,
                    help="Base output directory (overrides config)")
    ap.add_argument("--mask-pngs-dir", default=None,
                    help="Directory with DWI mask PNG slices (optional)")
    ap.add_argument("--adc-dir", default=None,
                    help="ADC raw data directory containing adc.nii.gz (optional)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip steps whose output already exists")

    args = ap.parse_args(cli_args)

    pid = args.patient
    if pid is None:
        ap.error("--patient is required")

    # Resolve DWI path – prefer raw (native-space) DWI for registration
    dwi = args.dwi
    if dwi is None:
        candidates = []
        # 1) Curated raw NIfTI (native DWI space, matches PNG masks)
        if CURATED_RAW_NIFTI_DIR is not None:
            candidates.append(os.path.join(
                CURATED_RAW_NIFTI_DIR,
                f"sub-stroke_{pid}", "ses-02",
                f"sub-stroke_{pid}_ses-02_dwi.nii.gz",
            ))
        # 2) Local preprocessed raw DWI
        if DWI_NIFTI_DIR is not None:
            candidates.append(os.path.join(
                DWI_NIFTI_DIR, f"sub-{pid}_ses-01", f"sub-{pid}_dwi.nii.gz",
            ))
        for c in candidates:
            if os.path.exists(c):
                dwi = c
                break
    if dwi is None:
        ap.error("--dwi is required (or provide --patient with DWI_NIFTI_DIR/CURATED_RAW_NIFTI_DIR in config)")

    # Resolve NCCT path
    ncct = args.ncct
    if ncct is None and NCCT_NIFTI_DIR is not None:
        ncct = os.path.join(NCCT_NIFTI_DIR, f"sub-{pid}_ses-01_ncct.nii.gz")
    if ncct is None:
        ap.error("--ncct is required (or provide --patient with NCCT_NIFTI_DIR in config)")

    # Resolve output dir
    output_dir = args.output_dir
    if output_dir is None and REG_ROOT is not None:
        output_dir = REG_ROOT
    if output_dir is None:
        ap.error("--output-dir is required (or set REG_ROOT in config)")

    # Resolve mask PNGs dir from DWI GT
    mask_pngs_dir = args.mask_pngs_dir
    if mask_pngs_dir is None and DWI_GT_ROOT is not None:
        candidate = os.path.join(DWI_GT_ROOT, f"CTP_{pid}")
        if os.path.isdir(candidate):
            mask_pngs_dir = candidate

    # Resolve ADC dir
    adc_dir = args.adc_dir
    if adc_dir is None and ADC_RAW_DIR is not None:
        candidate = os.path.join(ADC_RAW_DIR, f"sub-stroke{pid}_ses-01")
        if os.path.isdir(candidate):
            adc_dir = candidate

    register_dwi_to_ncct(
        pid=pid,
        dwi_path=dwi,
        ncct_path=ncct,
        output_dir=output_dir,
        mask_pngs_dir=mask_pngs_dir,
        adc_dir=adc_dir,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
