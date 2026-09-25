"""
s01_ctp_motion_correction.py
============================
CTP motion correction and temporal resampling.

Reads a Siemens CTP DICOM folder (interleaved Z*T frames), reconstructs
a 4-D volume, performs 3-D rigid registration of every time-point to t=0,
resamples the result to uniform 1 fps, and writes a geometry-correct 4-D
NIfTI file.

Standalone usage
----------------
    python s01_ctp_motion_correction.py \
        --dcm-dir /data/patient/DICOM/ctp_series \
        --output  /out/sub-001_ctp_mc_1fps.nii.gz

Pipeline usage
--------------
    from registration.s01_ctp_motion_correction import (
        process_ctp_motion_and_temporal_resample_sitk,
    )
"""

import os
import argparse

import numpy as np
import pydicom
from pydicom.misc import is_dicom
from collections import defaultdict
from tqdm import tqdm
import SimpleITK as sitk

try:
    from config import DICOM_ROOT, DICOM_HEADERS_CSV, CTP_MC_DIR
except ImportError:
    DICOM_ROOT = None
    DICOM_HEADERS_CSV = None
    CTP_MC_DIR = None


# -------------------------
# Helpers
# -------------------------

def dicom_time_to_seconds(tm):
    """DICOM TM: HHMMSS.frac -> seconds since midnight (float)."""
    if tm is None or tm == "" or (isinstance(tm, float) and np.isnan(tm)):
        return None
    tm = str(tm)
    if "." in tm:
        main, frac = tm.split(".", 1)
        frac = float("0." + frac)
    else:
        main, frac = tm, 0.0
    main = main.zfill(6)
    hh, mm, ss = int(main[:2]), int(main[2:4]), int(main[4:6])
    return hh * 3600 + mm * 60 + ss + float(frac)


def build_assumed_time_vector(T=30, split=20, dt1=1.0, dt2=2.0):
    """
    Your known scheme:
      frames 0..19: 1s apart
      frames 20..29: 2s apart
    Returns t_in (T,).
    """
    t = [0.0]
    for i in range(1, T):
        t.append(t[-1] + (dt1 if i < split else dt2))
    return np.array(t, dtype=np.float32)


def z_bin(z, z_bin_mm=0.5):
    """Bin z (IPP[2]) to stabilize float jitter."""
    return float(np.round(float(z) / z_bin_mm) * z_bin_mm)


# -------------------------
# 1) Robust Siemens CTP DICOM -> 4D numpy (T,Z,Y,X)
# -------------------------

def dicom_ctp_to_4d(dcm_dir, Z_expected=14, z_bin_mm=0.5):
    """
    Siemens CTP often stored as interleaved frames: total = Z*T (e.g. 14*30=420).
    We reconstruct 4D using z positions.

    Returns:
      ctp_tzyx: (T,Z,Y,X) int16
      z_keys: sorted unique z bins (length Z)
      by_z: dict z -> list of frames sorted by (time, instance)
      ref_ds: one representative pydicom ds (for PixelSpacing, IOP, IPP)
    """
    frames = []
    ref_ds = None

    for fname in os.listdir(dcm_dir):
        fp = os.path.join(dcm_dir, fname)
        if not os.path.isfile(fp):
            continue
        try:
            if not is_dicom(fp):
                continue
            ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)

            ipp = getattr(ds, "ImagePositionPatient", None)
            if ipp is None:
                continue

            zb = z_bin(float(ipp[2]), z_bin_mm)

            t = dicom_time_to_seconds(getattr(ds, "AcquisitionTime", None))
            inst = int(getattr(ds, "InstanceNumber", 0))

            frames.append({"file": fp, "z": zb, "t": t, "inst": inst})

            if ref_ds is None:
                ref_ds = ds
        except Exception:
            continue

    if not frames:
        raise RuntimeError(f"No DICOM frames found in: {dcm_dir}")

    z_keys = sorted(set(f["z"] for f in frames))
    if Z_expected is not None and len(z_keys) != Z_expected:
        raise RuntimeError(
            f"Expected Z={Z_expected} unique z slices, but found {len(z_keys)}. "
            f"Try adjusting z_bin_mm (currently {z_bin_mm}).")
    print(f"  Auto-detected Z={len(z_keys)} unique z slices from CTP DICOM.")

    by_z = defaultdict(list)
    for f in frames:
        by_z[f["z"]].append(f)

    T = len(frames) // len(z_keys)
    if T * len(z_keys) != len(frames):
        raise RuntimeError(
            f"Total frames ({len(frames)}) not divisible by Z ({len(z_keys)}). "
            "Series may be mixed, or z_bin_mm too strict/loose."
        )

    def sort_key(f):
        return (f["t"] if f["t"] is not None else 1e18, f["inst"])

    for z in z_keys:
        by_z[z].sort(key=sort_key)
        if len(by_z[z]) != T:
            print(f"[WARN] z={z} has {len(by_z[z])} frames (expected {T}).")

    # Read first pixel to get Y,X
    ds0 = pydicom.dcmread(by_z[z_keys[0]][0]["file"], force=True)
    Y, X = int(ds0.Rows), int(ds0.Columns)

    ctp_tzyx = np.zeros((T, len(z_keys), Y, X), dtype=np.int16)

    for zi, z in enumerate(z_keys):
        if len(by_z[z]) < T:
            raise RuntimeError(f"Not enough frames at z={z} to fill T={T}.")
        for ti in range(T):
            fp = by_z[z][ti]["file"]
            ds = pydicom.dcmread(fp, force=True)

            img = ds.pixel_array.astype(np.float32)
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            img = img * slope + intercept

            ctp_tzyx[ti, zi] = img.astype(np.int16)

    return ctp_tzyx, z_keys, by_z, ref_ds


# -------------------------
# 2) Build a geometry-correct 3D reference SITK image (one timepoint)
# -------------------------

def sitk_ref3d_from_one_timepoint(by_z, z_keys, ti=0):
    """
    Reads a single timepoint (ti) as a proper 3D image using ImageSeriesReader.
    This avoids non-uniform sampling warnings from reading all 420 frames at once.
    """
    reader = sitk.ImageSeriesReader()
    files_t = [by_z[z][ti]["file"] for z in z_keys]  # sorted by z_keys
    reader.SetFileNames(files_t)
    img3d = reader.Execute()
    return img3d


# -------------------------
# 3) Apply correct 3D geometry to a 4D SITK image
# -------------------------

def apply_geometry_to_4d(img4d, ref3d, z_keys, st=1.0):
    """
    img4d: sitk.Image 4D (x,y,z,t) created from array (T,Z,Y,X) using GetImageFromArray(..., isVector=False)
    ref3d: sitk.Image 3D with correct origin/spacing/direction
    z_keys: used to compute robust dz (median diff)
    st: time spacing in seconds (uniform; store true t vector separately if non-uniform)
    """
    # origin
    ox, oy, oz = ref3d.GetOrigin()
    img4d.SetOrigin((ox, oy, oz, 0.0))

    # direction: embed 3D into 4D (block diagonal)
    d3 = np.array(ref3d.GetDirection(), dtype=float).reshape(3, 3)
    d4 = np.eye(4, dtype=float)
    d4[:3, :3] = d3
    img4d.SetDirection(tuple(d4.flatten()))

    # spacing: dx,dy from ref3d; dz from z_keys
    sx, sy, _ = ref3d.GetSpacing()
    dz = float(np.median(np.diff(sorted(z_keys)))) if len(z_keys) > 1 else float(ref3d.GetSpacing()[2])
    img4d.SetSpacing((sx, sy, dz, float(st)))

    return img4d


# -------------------------
# 4) Motion correction (3D rigid) per timepoint to t=0 using SimpleITK
# -------------------------

def motion_correct_ctp_3d_rigid_sitk(ctp_tzyx, ref3d_geometry, use_smoothing=True):
    """
    ctp_tzyx: numpy (T,Z,Y,X) float32/int
    ref3d_geometry: a 3D sitk.Image whose geometry we want for volumes (origin/spacing/direction)
    Returns:
      mc_tzyx: numpy (T,Z,Y,X) float32 motion corrected (registered to t=0)
      transforms: list of sitk.Transform per timepoint
    """
    T, Z, Y, X = ctp_tzyx.shape

    def to_sitk_3d(vol_zyx):
        img = sitk.GetImageFromArray(vol_zyx.astype(np.float32))  # (Z,Y,X)
        img.CopyInformation(ref3d_geometry)  # copy origin/spacing/direction
        return img

    fixed = to_sitk_3d(ctp_tzyx[0])
    if use_smoothing:
        fixed_f = sitk.DiscreteGaussian(fixed, variance=1.0)
    else:
        fixed_f = fixed

    mc = np.zeros((T, Z, Y, X), dtype=np.float32)
    mc[0] = ctp_tzyx[0].astype(np.float32)

    transforms = [sitk.Euler3DTransform()]  # identity-ish placeholder for t=0

    # Configure registration method once
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2, seed=42)
    reg.SetInterpolator(sitk.sitkLinear)

    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=1e-3,
        numberOfIterations=300,
        gradientMagnitudeTolerance=1e-6
    )
    reg.SetOptimizerScalesFromPhysicalShift()

    for t in tqdm(range(1, T), desc="Rigid motion correction (3D)"):
        moving = to_sitk_3d(ctp_tzyx[t])
        moving_f = sitk.DiscreteGaussian(moving, variance=1.0) if use_smoothing else moving

        init_tx = sitk.CenteredTransformInitializer(
            fixed_f, moving_f,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        reg.SetInitialTransform(init_tx, inPlace=False)

        final_tx = reg.Execute(fixed_f, moving_f)

        warped = sitk.Resample(
            moving, fixed, final_tx,
            sitk.sitkLinear, 0.0, sitk.sitkFloat32
        )

        mc[t] = sitk.GetArrayFromImage(warped)  # back to (Z,Y,X)
        transforms.append(final_tx)

    return mc, transforms


# -------------------------
# 5) Temporal resampling to 1 fps (linear interpolation in time)
# -------------------------

def temporal_resample_to_1fps(ctp_tzyx_float, t_in):
    """
    ctp_tzyx_float: (T,Z,Y,X) float32
    t_in: (T,) seconds (nonuniform)
    Returns:
      out: (Tout,Z,Y,X) float32 at 1 fps
      t_out: uniform time vector (seconds)
    """
    T, Z, Y, X = ctp_tzyx_float.shape
    t0, tN = float(t_in[0]), float(t_in[-1])
    t_out = np.arange(t0, np.floor(tN) + 1.0, 1.0, dtype=np.float32)

    out = np.zeros((len(t_out), Z, Y, X), dtype=np.float32)

    # Efficient reshape: (T, Nvox)
    flat = ctp_tzyx_float.reshape(T, -1)
    out_flat = np.zeros((len(t_out), flat.shape[1]), dtype=np.float32)

    for i in tqdm(range(flat.shape[1]), desc="Temporal interpolation (voxels)"):
        out_flat[:, i] = np.interp(t_out, t_in, flat[:, i])

    out = out_flat.reshape(len(t_out), Z, Y, X)
    return out, t_out


# -------------------------
# 6) Save 4D as NIfTI using SimpleITK (with geometry)
# -------------------------

def save_4d_nifti_sitk(ctp_tzyx, ref3d, z_keys, out_path, st=1.0):
    """
    ctp_tzyx: numpy (T,Z,Y,X)
    ref3d: sitk 3D image with correct geometry
    """
    img4d = sitk.GetImageFromArray(ctp_tzyx.astype(np.float32), isVector=False)  # (x,y,z,t)
    img4d = apply_geometry_to_4d(img4d, ref3d, z_keys, st=st)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sitk.WriteImage(img4d, out_path)
    return img4d


# -------------------------
# Full pipeline
# -------------------------

def process_ctp_motion_and_temporal_resample_sitk(
    dcm_dir,
    out_nifti_path,
    Z_expected=14,
    z_bin_mm=0.5,
    split=20,
    dt1=1.0,
    dt2=2.0,
    use_smoothing=True
):
    # Step 1: DICOM -> 4D array (T,Z,Y,X)
    ctp_tzyx, z_keys, by_z, ref_ds = dicom_ctp_to_4d(dcm_dir, Z_expected=Z_expected, z_bin_mm=z_bin_mm)
    T, Z, Y, X = ctp_tzyx.shape
    print("Loaded CTP (T,Z,Y,X):", ctp_tzyx.shape)

    # Step 2: 3D ref geometry from one timepoint
    ref3d = sitk_ref3d_from_one_timepoint(by_z, z_keys, ti=0)

    # Step 3: Motion correction (3D rigid to t=0)
    ctp_mc, transforms = motion_correct_ctp_3d_rigid_sitk(
        ctp_tzyx.astype(np.float32),
        ref3d_geometry=ref3d,
        use_smoothing=use_smoothing
    )
    print("Motion corrected:", ctp_mc.shape)

    # Step 4: Temporal resampling to 1 fps using known timing
    t_in = build_assumed_time_vector(T=T, split=split, dt1=dt1, dt2=dt2)
    ctp_1fps, t_out = temporal_resample_to_1fps(ctp_mc, t_in)
    print("Temporal resampled:", ctp_1fps.shape, "t_out last:", t_out[-1])

    # Step 5: Save (time spacing is 1.0s; true nonuniform timing stored separately if needed)
    img4d = save_4d_nifti_sitk(ctp_1fps, ref3d, z_keys, out_nifti_path, st=1.0)

    print("Saved:", out_nifti_path)
    print("Final SITK size:", img4d.GetSize(), "spacing:", img4d.GetSpacing())
    return img4d


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _find_ctp_dcm_dir_from_csv(pid, csv_path):
    """Look up the CTP DICOM folder for *pid* from the headers CSV.

    Finds rows where PatientID matches CTP_{pid} and picks the folder
    with the most DICOM files (the raw CTP acquisition).
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    patient_id = f"CTP_{pid}"
    matches = df[df["PatientID"].astype(str) == patient_id]
    if matches.empty:
        return None
    # Pick the folder with the largest number of DICOM files
    best = matches.loc[matches["NumDicomFiles"].idxmax()]
    return str(best["Folder"])


def main(cli_args=None):
    parser = argparse.ArgumentParser(
        description="CTP motion correction (3-D rigid to t=0) and temporal resampling to 1 fps."
    )
    parser.add_argument(
        "--patient", "--pid", default=None,
        help="Patient ID (e.g. 01_001). Resolves DICOM dir and output from config."
    )
    parser.add_argument(
        "--dcm-dir", default=None,
        help="Path to the DICOM folder containing the CTP series (overrides config lookup)."
    )
    parser.add_argument(
        "--output", default=None,
        help="Output NIfTI path (overrides config)."
    )
    parser.add_argument(
        "--csv-path", default=DICOM_HEADERS_CSV,
        help="DICOM headers CSV for auto-resolving --dcm-dir (default: from config)."
    )
    parser.add_argument(
        "--z-expected", type=int, default=14,
        help="Expected number of unique z slices (default: 14)."
    )
    parser.add_argument(
        "--z-bin-mm", type=float, default=0.5,
        help="Bin width in mm for grouping z positions (default: 0.5)."
    )
    parser.add_argument(
        "--split", type=int, default=20,
        help="Frame index where temporal spacing changes from dt1 to dt2 (default: 20)."
    )
    parser.add_argument(
        "--dt1", type=float, default=1.0,
        help="Time step (s) for frames before --split (default: 1.0)."
    )
    parser.add_argument(
        "--dt2", type=float, default=2.0,
        help="Time step (s) for frames from --split onward (default: 2.0)."
    )
    parser.add_argument(
        "--no-smoothing", action="store_true",
        help="Disable Gaussian smoothing before registration."
    )
    args = parser.parse_args(cli_args)

    pid = args.patient

    # Resolve dcm-dir
    dcm_dir = args.dcm_dir
    if dcm_dir is None and pid is not None and args.csv_path is not None:
        dcm_dir = _find_ctp_dcm_dir_from_csv(pid, args.csv_path)
    if dcm_dir is None:
        parser.error("--dcm-dir is required (or provide --patient with a valid CSV)")

    # Resolve output
    output = args.output
    if output is None and pid is not None and CTP_MC_DIR is not None:
        output = os.path.join(CTP_MC_DIR, f"sub-{pid}_ses-01_ctp_mc_1fps.nii.gz")
    if output is None:
        parser.error("--output is required (or provide --patient with CTP_MC_DIR in config)")

    process_ctp_motion_and_temporal_resample_sitk(
        dcm_dir=dcm_dir,
        out_nifti_path=output,
        Z_expected=args.z_expected,
        z_bin_mm=args.z_bin_mm,
        split=args.split,
        dt1=args.dt1,
        dt2=args.dt2,
        use_smoothing=not args.no_smoothing,
    )


if __name__ == "__main__":
    main()
