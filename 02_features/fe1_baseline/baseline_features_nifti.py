"""
Baseline feature extraction for NIfTI CTP volumes with 6-class segmentation masks.

Extracts statistical features (mean, std, skewness, min, max, kurtosis) from 3x3xT
spatiotemporal kernels for each of 6 tissue classes across CTP volumes.

Input format:
    - CTP: NIfTI 4D volumes (T, Z, Y, X) with 40 timepoints, values in [0, 100]
    - Labels: NIfTI 3D volumes (Z, Y, X) with values {0, 1, 2, 3, 4, 5, 6}

Classes:
    1. core_fi
    2. core_brain
    3. pen_fi
    4. pen_brain
    5. clb_brain
    6. NHB_fi

Only processes slices where:
    - Penumbra (class 3 or 4) is present
    - CTP has non-zero signal
"""

import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
from scipy.stats import skew, kurtosis
import argparse
import warnings

warnings.filterwarnings('ignore')


class BaselineFeaturesNIfTI:
    """Extract baseline features from NIfTI CTP volumes using 6-class masks."""
    
    # Class definitions
    CLASS_NAMES = [
        "background",
        "core_fi",
        "core_brain", 
        "pen_fi",
        "pen_brain",
        "clb_brain",
        "NHB_fi"
    ]
    
    PENUMBRA_CLASSES = [3, 4]  # pen_fi, pen_brain
    
    def __init__(self, derivatives_dir, output_dir=None):
        """
        Initialize feature extractor.
        
        Parameters
        ----------
        derivatives_dir : str or Path
            Root derivatives folder (contains sub-stroke_XX_XXX/ subdirs)
        output_dir : str or Path, optional
            Output directory for results
        """
        self.derivatives_dir = Path(derivatives_dir)
        
        if output_dir is None:
            output_dir = Path(__file__).parent / "results"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_nifti(self, path):
        """Load NIfTI file and return numpy array."""
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)
    
    def find_valid_slices(self, ctp_volume, label_volume, min_overlap_ratio=0.5):
        """
        Find slices with both CTP signal and penumbra, ensuring adequate overlap.
        
        A slice is valid only if a sufficient fraction of labelled (non-background)
        voxels actually have CTP signal.  This filters out "half slices" where
        the label mask extends beyond the CTP field-of-view.
        
        Parameters
        ----------
        ctp_volume : ndarray (T, Z, Y, X)
            CTP volume
        label_volume : ndarray (Z, Y, X)
            Label volume
        min_overlap_ratio : float
            Minimum fraction of labelled voxels that must have CTP signal
            (default 0.5 = 50 %).  Set to 0 to disable.
            
        Returns
        -------
        valid_slices : list of int
            Slice indices to process
        """
        T, Z, Y, X = ctp_volume.shape
        
        valid_slices = []
        for z in range(Z):
            label_slice = label_volume[z, :, :]
            has_penumbra = np.isin(label_slice, self.PENUMBRA_CLASSES).any()
            if not has_penumbra:
                continue
            
            # Mask of all labelled (non-background) voxels
            labelled_mask = label_slice > 0
            n_labelled = labelled_mask.sum()
            if n_labelled == 0:
                continue
            
            # CTP coverage mask: voxel has signal if any timepoint is non-zero
            ctp_slice = ctp_volume[:, z, :, :]           # (T, Y, X)
            ctp_signal_mask = np.any(ctp_slice != 0, axis=0)  # (Y, X)
            
            # Overlap: labelled voxels that also have CTP signal
            overlap = (labelled_mask & ctp_signal_mask).sum()
            overlap_ratio = overlap / n_labelled
            
            if overlap_ratio >= min_overlap_ratio:
                valid_slices.append(z)
            else:
                print(f"    Slice {z}: skipped – CTP covers only "
                      f"{overlap}/{n_labelled} labelled voxels "
                      f"({overlap_ratio:.1%} < {min_overlap_ratio:.0%})")
        
        return valid_slices
    
    def extract_features_from_kernel(self, kernel):
        """
        Extract 6 statistical features from a 3D kernel.
        
        Parameters
        ----------
        kernel : ndarray (3, 3, T)
            Spatiotemporal kernel
            
        Returns
        -------
        features : dict
            Dictionary with feature names and values
        """
        kernel_flat = kernel.flatten()
        
        features = {}
        
        # Basic statistics
        features['mean'] = np.mean(kernel_flat)
        features['std'] = np.std(kernel_flat)
        features['min'] = np.min(kernel_flat)
        features['max'] = np.max(kernel_flat)
        
        # Handle skewness and kurtosis
        std_val = features['std']
        
        if std_val < 1e-10:  # Zero variance
            features['skewness'] = 0.0
            features['kurtosis'] = 0.0
        else:
            try:
                skew_val = skew(kernel_flat, nan_policy='omit')
                features['skewness'] = 0.0 if np.isnan(skew_val) or np.isinf(skew_val) else skew_val
            except:
                features['skewness'] = 0.0
            
            try:
                kurt_val = kurtosis(kernel_flat, nan_policy='omit')
                features['kurtosis'] = 0.0 if np.isnan(kurt_val) or np.isinf(kurt_val) else kurt_val
            except:
                features['kurtosis'] = 0.0
        
        return features
    
    def extract_slice_features(self, ctp_slice, label_slice, slice_idx):
        """
        Extract features for all classes in a single slice.
        
        Parameters
        ----------
        ctp_slice : ndarray (T, Y, X)
            CTP data for one slice
        label_slice : ndarray (Y, X)
            Label mask for one slice
        slice_idx : int
            Slice index
            
        Returns
        -------
        slice_features : list of dict
            Features for each class present in the slice
        """
        T, Y, X = ctp_slice.shape
        kernel_size = (3, 3, T)
        
        # Pad spatial dimensions only
        pad_h = 1
        pad_w = 1
        ctp_padded = np.pad(ctp_slice, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        label_padded = np.pad(label_slice, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        
        # Initialize feature storage per class
        class_features = {cls: [] for cls in range(1, 7)}  # Classes 1-6
        
        # Pre-compute per-voxel CTP signal mask (has signal at any timepoint)
        ctp_signal_mask = np.any(ctp_slice != 0, axis=0)   # (Y, X)
        # Pad the signal mask to match ctp_padded coordinates
        signal_mask_padded = np.pad(ctp_signal_mask, ((pad_h, pad_h), (pad_w, pad_w)),
                                     mode='constant', constant_values=False)
        skipped_no_ctp = 0
        skipped_boundary = 0
        
        # Extract features for each voxel
        for y in range(Y):
            for x in range(X):
                label_val = int(label_slice[y, x])
                
                # Skip background (class 0)
                if label_val == 0:
                    continue
                
                # Skip voxels where the CTP has no signal (half-slice artefact)
                if not ctp_signal_mask[y, x]:
                    skipped_no_ctp += 1
                    continue
                
                # Skip boundary voxels: if any of the 9 neighbors in the
                # 3×3 kernel have zero CTP signal, the kernel straddles the
                # brain edge and will produce unrealistic feature values.
                kernel_mask = signal_mask_padded[y:y+3, x:x+3]  # (3, 3)
                if not kernel_mask.all():
                    skipped_boundary += 1
                    continue
                
                # Extract 3x3xT kernel
                kernel = ctp_padded[:, y:y+3, x:x+3]  # Shape: (T, 3, 3)
                kernel = kernel.transpose(1, 2, 0)    # Shape: (3, 3, T)
                
                if kernel.shape == kernel_size:
                    features = self.extract_features_from_kernel(kernel)
                    features['y'] = y
                    features['x'] = x
                    features['slice'] = slice_idx
                    features['class'] = label_val
                    features['class_name'] = self.CLASS_NAMES[label_val]
                    
                    class_features[label_val].append(features)
        
        if skipped_no_ctp > 0 or skipped_boundary > 0:
            print(f"      Slice {slice_idx}: skipped {skipped_no_ctp} no-CTP + "
                  f"{skipped_boundary} boundary voxels")
        
        # Flatten to single list
        all_features = []
        for cls in range(1, 7):
            all_features.extend(class_features[cls])
        
        return all_features
    
    def compute_class_statistics(self, df_features):
        """
        Compute mean and std statistics per class and slice.
        
        Parameters
        ----------
        df_features : DataFrame
            Features for all voxels
            
        Returns
        -------
        df_stats : DataFrame
            Aggregated statistics per class and slice
        """
        # Group by slice and class
        feature_cols = ['mean', 'std', 'skewness', 'min', 'max', 'kurtosis']
        
        df_stats = df_features.groupby(['slice', 'class', 'class_name'])[feature_cols].agg(
            ['mean', 'std', 'min', 'max', 'median']
        ).reset_index()
        
        # Flatten multi-level column names
        df_stats.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                           for col in df_stats.columns.values]
        
        # Add voxel count
        voxel_counts = df_features.groupby(['slice', 'class', 'class_name']).size().reset_index(name='voxel_count')
        df_stats = df_stats.merge(voxel_counts, on=['slice', 'class', 'class_name'])
        
        return df_stats
    
    def process_patient(self, patient_id, save_results=True):
        """
        Process a single patient.
        
        Parameters
        ----------
        patient_id : str
            Patient identifier (e.g., "sub-stroke_01_001")
        save_results : bool
            Whether to save results to CSV
            
        Returns
        -------
        df_features : DataFrame
            Per-voxel features
        df_stats : DataFrame
            Aggregated statistics per class
        """
        print(f"\n{'='*60}")
        print(f"Processing patient: {patient_id}")
        print(f"{'='*60}")
        
        ses_dir = self.derivatives_dir / patient_id / "ses-01"
        ctp_path = ses_dir / f"{patient_id}_ses-01_space-ncct_ctp.nii.gz"
        label_path = ses_dir / f"{patient_id}_space-ncct_bi-temporal_lesion_msk.nii.gz"
        
        if not ctp_path.exists():
            raise FileNotFoundError(f"No CTP file found: {ctp_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"No label file found: {label_path}")
        
        print(f"CTP file:   {ctp_path.name}")
        print(f"Label file: {label_path.name}")
        
        # Load volumes
        print("Loading volumes...")
        ctp_volume = self.load_nifti(ctp_path)      # (T, Z, Y, X)
        label_volume = self.load_nifti(label_path).astype(np.int32)  # (Z, Y, X)
        
        print(f"  CTP shape:   {ctp_volume.shape}")
        print(f"  Label shape: {label_volume.shape}")
        print(f"  CTP range:   [{ctp_volume.min():.2f}, {ctp_volume.max():.2f}]")
        
        unique_classes = np.unique(label_volume)
        print(f"  Classes present: {unique_classes}")
        for cls in unique_classes:
            if cls > 0:
                count = (label_volume == cls).sum()
                print(f"    Class {cls} ({self.CLASS_NAMES[int(cls)]}): {count:,} voxels")
        
        # Find valid slices
        print("\nFinding valid slices (with CTP signal and penumbra)...")
        valid_slices = self.find_valid_slices(ctp_volume, label_volume)
        print(f"  Valid slices: {len(valid_slices)} / {label_volume.shape[0]}")
        print(f"  Slice indices: {valid_slices}")
        
        if len(valid_slices) == 0:
            print("  WARNING: No valid slices found!")
            return pd.DataFrame(), pd.DataFrame()
        
        # Extract features for each valid slice
        print("\nExtracting features...")
        all_features = []
        
        for z in tqdm(valid_slices, desc="Processing slices"):
            ctp_slice = ctp_volume[:, z, :, :]     # (T, Y, X)
            label_slice = label_volume[z, :, :]    # (Y, X)
            
            slice_features = self.extract_slice_features(ctp_slice, label_slice, z)
            all_features.extend(slice_features)
        
        # Convert to DataFrame
        df_features = pd.DataFrame(all_features)
        print(f"\nExtracted {len(df_features):,} voxel features")
        
        # Compute aggregated statistics
        print("Computing class statistics...")
        df_stats = self.compute_class_statistics(df_features)
        print(f"  Statistics computed for {len(df_stats)} class-slice combinations")
        
        # Save results
        if save_results:
            patient_output_dir = self.output_dir / patient_id
            patient_output_dir.mkdir(parents=True, exist_ok=True)
            
            features_path = patient_output_dir / f"{patient_id}_voxel_features.csv"
            stats_path = patient_output_dir / f"{patient_id}_class_statistics.csv"
            
            df_features.to_csv(features_path, index=False)
            df_stats.to_csv(stats_path, index=False)
            
            print(f"\nResults saved:")
            print(f"  Voxel features: {features_path}")
            print(f"  Class stats:    {stats_path}")
        
        return df_features, df_stats


def find_patient_pairs(derivatives_dir):
    """
    Find patients with both CTP and 6-class label in derivatives/sub-stroke_*/ses-01/.
    
    Returns
    -------
    patient_ids : list of str
        List of patient identifiers (e.g., "sub-stroke_01_001")
    """
    import re
    derivatives_dir = Path(derivatives_dir)
    
    patient_ids = []
    for subdir in sorted(derivatives_dir.iterdir()):
        if not subdir.is_dir():
            continue
        m = re.match(r"(sub-stroke_\d{2}_\d{3})", subdir.name)
        if m is None:
            continue
        pid = m.group(1)
        ses_dir = subdir / "ses-01"
        if not ses_dir.is_dir():
            continue
        ctp_path = ses_dir / f"{pid}_ses-01_space-ncct_ctp.nii.gz"
        label_path = ses_dir / f"{pid}_space-ncct_bi-temporal_lesion_msk.nii.gz"
        if ctp_path.exists() and label_path.exists():
            patient_ids.append(pid)
    
    return patient_ids


def main():
    """Main entry point for batch processing."""
    parser = argparse.ArgumentParser(
        description="Extract baseline features from NIfTI CTP volumes with 6-class masks."
    )
    
    CURATED = os.environ.get("SUS_CURATED_ROOT", "/path/to/Curated-SUS-2025")
    
    parser.add_argument(
        "--derivatives-dir",
        type=str,
        default=os.path.join(CURATED, "derivatives"),
        help="Root derivatives folder (contains sub-stroke_XX_XXX/ subdirs)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.environ.get("FEATURE_WORK_ROOT", "/path/to/feature-work"), "baseline_features_nifti", "results"),
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--patients",
        type=str,
        default=None,
        help="Comma-separated list of patient IDs (e.g., 'sub-01_001,sub-01_002'). Process all if not specified."
    )
    
    parser.add_argument(
        "--start-idx",
        type=int,
        default=None,
        help="Start index (1-based, inclusive)"
    )
    
    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="End index (1-based, inclusive)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List patients and exit without processing"
    )
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = BaselineFeaturesNIfTI(
        derivatives_dir=args.derivatives_dir,
        output_dir=args.output_dir
    )
    
    # Find all patient pairs
    print("Scanning for patient files...")
    all_patient_ids = find_patient_pairs(args.derivatives_dir)
    print(f"Found {len(all_patient_ids)} patients with matching CTP and label files")
    
    if len(all_patient_ids) == 0:
        print("ERROR: No matching patient files found!")
        return
    
    # Select patients to process
    if args.patients:
        # User-specified list
        selected_ids = [p.strip() for p in args.patients.split(',') if p.strip()]
        selected_ids = [p for p in selected_ids if p in all_patient_ids]
        
        missing = set(args.patients.split(',')) - set(selected_ids)
        if missing:
            print(f"Warning: These patients were not found: {missing}")
    else:
        # Range selection
        start = (args.start_idx - 1) if args.start_idx else 0
        end = args.end_idx if args.end_idx else len(all_patient_ids)
        selected_ids = all_patient_ids[start:end]
    
    print(f"\nSelected {len(selected_ids)} patients to process:")
    for i, pid in enumerate(selected_ids, 1):
        print(f"  {i:2d}. {pid}")
    
    if args.dry_run:
        print("\n(Dry run - exiting)")
        return
    
    # Process each patient
    print(f"\n{'='*60}")
    print("Starting batch processing")
    print(f"{'='*60}\n")
    
    success_count = 0
    error_count = 0
    
    for i, patient_id in enumerate(selected_ids, 1):
        print(f"\n[{i}/{len(selected_ids)}]")
        
        try:
            df_features, df_stats = processor.process_patient(
                patient_id=patient_id,
                save_results=True
            )
            
            if len(df_features) > 0:
                success_count += 1
                print(f"✅ Successfully processed {patient_id}")
            else:
                print(f"⚠️  No features extracted for {patient_id}")
        
        except Exception as e:
            error_count += 1
            print(f"❌ Error processing {patient_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*60}")
    print("Batch processing completed")
    print(f"{'='*60}")
    print(f"Successful: {success_count}")
    print(f"Errors:     {error_count}")
    print(f"Total:      {len(selected_ids)}")
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
