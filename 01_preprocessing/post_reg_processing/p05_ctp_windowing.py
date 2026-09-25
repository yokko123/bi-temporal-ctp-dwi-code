"""
Post-registration step 5 – Apply HU windowing to 4D CTP volumes.

Clips intensities to [low, high] and optionally normalises to [0, 1]
or converts to [0, 255] gray-level.  The 4D dimensionality of the
image is preserved throughout.

Original script
---------------
    SUS2025-Preprocessing/scripts/ctp_windowing/ctp_windowing.py
"""

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk

# ─── Config defaults (optional) ─────────────────────────────────────
try:
    from config import CTP_SS_DIR, CTP_WINDOWED_DIR
except ImportError:
    CTP_SS_DIR = None
    CTP_WINDOWED_DIR = None


# =====================================================================
# Core processing function
# =====================================================================

def window_ctp(input_path, output_path, low=0.0, high=100.0,
               normalize=False, gray_level=False, verbose=False):
    """Apply HU windowing to a single 4D CTP NIfTI volume."""

    # 1. Read the original image
    img = sitk.ReadImage(str(input_path))

    # 2. Convert to numpy
    # Note: SITK reads 4D as (Time, Z, Y, X)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)

    if verbose:
        brain = arr[arr > 0]
        if brain.size > 0:
            print(f"  Input Shape: {img.GetSize()} | Dimensions: {img.GetDimension()}")
            print("  Original percentiles:", np.percentile(brain, [1, 5, 50, 95, 99]))

    # 3. Clip intensities
    arr = np.clip(arr, low, high)

    # 4. Optional normalization / gray-level conversion
    if gray_level:
        arr = ((arr - low) / (high - low) * 255.0).astype(np.uint8).astype(np.float32)
    elif normalize:
        arr = (arr - low) / (high - low)

    # 5. Convert back to SITK image
    # IMPORTANT: isVector=False ensures it stays a 4D image (T, Z, Y, X)
    # rather than a 3D image with T-channels.
    img_windowed = sitk.GetImageFromArray(arr, isVector=False)

    # 6. Copy Metadata
    # Now that both are confirmed 4D, CopyInformation will work perfectly
    img_windowed.CopyInformation(img)

    if verbose:
        print("  Shape check -> before:", img.GetSize(), "after:", img_windowed.GetSize())

    # 7. Save
    sitk.WriteImage(img_windowed, str(output_path), useCompression=True)

    if verbose:
        brain_w = arr[arr > 0]
        if brain_w.size > 0:
            print("  Windowed percentiles:", np.percentile(brain_w, [1, 5, 50, 95, 99]))


# =====================================================================
# CLI entry-point
# =====================================================================

def main(args=None):
    """Run CTP windowing on a directory of 4D NIfTI files."""

    parser = argparse.ArgumentParser(
        description="Window 4D CTP volumes safely (preserve 4D dimension)"
    )
    parser.add_argument("--input-dir", type=Path, default=CTP_SS_DIR,
                        help="Directory with input CTP NIfTI files")
    parser.add_argument("--output-dir", type=Path, default=CTP_WINDOWED_DIR,
                        help="Directory for windowed output files")
    parser.add_argument("--low", type=float, default=0.0,
                        help="Lower HU clipping bound (default: 0.0)")
    parser.add_argument("--high", type=float, default=100.0,
                        help="Upper HU clipping bound (default: 100.0)")
    parser.add_argument("--normalize", action="store_true",
                        help="Normalize to [0, 1] float range")
    parser.add_argument("--gray-level", action="store_true",
                        help="Convert HU to [0, 255] gray level (uint8)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output files (default: skip)")

    args = parser.parse_args(args)

    if args.input_dir is None or args.output_dir is None:
        parser.error("--input-dir and --output-dir are required "
                     "(no config defaults available)")

    args.input_dir = Path(args.input_dir)
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Using glob for nii.gz files
    files = sorted(args.input_dir.glob("*.nii.gz"))

    print(f"Found {len(files)} CTP files")

    for f in files:
        out_path = args.output_dir / f.name
        if out_path.exists() and not args.overwrite:
            print(f"\nSkipping {f.name} (already exists)")
            continue
        print(f"\nProcessing {f.name}")
        try:
            window_ctp(
                input_path=f,
                output_path=out_path,
                low=args.low,
                high=args.high,
                normalize=args.normalize,
                gray_level=args.gray_level,
                verbose=args.verbose,
            )
        except Exception as e:
            print(f"  ❌ Error: {e}")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
