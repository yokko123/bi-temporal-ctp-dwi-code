"""
Shared configuration for the SUS2025 preprocessing pipeline.

All default paths are defined here so they can be changed in one place.
Individual scripts accept CLI overrides for most paths.
"""

import os

# ─── Root directories ───────────────────────────────────────────────
# Raw DICOM data (read-only)
DICOM_ROOT = os.environ.get("SUS_DICOM_ROOT", "/path/to/SUS-2025-10-dicom")

# CTP ground truth labels (TIFF slices per patient)
CTP_GT_ROOT = os.environ.get("SUS_CTP_GT_ROOT", "/path/to/CTP-ground-truth")

# DWI ground truth masks (PNG slices per patient)
DWI_GT_ROOT = os.environ.get("SUS_DWI_GT_ROOT", "/path/to/DWI-ground-truth")

# CTP pre-processed masks (PNG slices, pre-registration native space)
CTP_NATIVE_MASKS = os.environ.get("SUS_CTP_NATIVE_MASKS", "/path/to/CTP-native-brain-masks")

# CTP parametric maps (DICOM, pre-computed by scanner)
CTP_PARAM_MAPS_ROOT = os.environ.get("SUS_CTP_PARAM_MAPS", "/path/to/CTP-parametric-maps")

# Curated dataset (read-only) – raw NIfTI files
CURATED_RAW_NIFTI_DIR = os.path.join(os.environ.get("SUS_CURATED_ROOT", "/path/to/Curated-SUS-2025"), "raw_data_nifti")

# ─── Working directories (outputs) ──────────────────────────────────
REG_ROOT = os.environ.get("SUS_WORK_ROOT", "/path/to/sus-work")
SUS_ROOT = os.environ.get("SUS_SCRATCH_ROOT", "/path/to/sus-scratch")

# DICOM headers (CSV per patient)
DICOM_HEADERS_DIR = os.path.join(REG_ROOT, "dicom_headers_sus_2025")
DICOM_HEADERS_CSV = os.path.join(DICOM_HEADERS_DIR, "all_dicom_headers_summary.csv")

# Step 1: Motion-corrected CTP (4D NIfTI, 1 fps)
CTP_MC_DIR = os.path.join(REG_ROOT, "motion_corrected_1fps")

# Step 2: NCCT NIfTI
NCCT_NIFTI_DIR = os.path.join(SUS_ROOT, "ncct_nifti")

# Step 3: CTP registered to NCCT space
CTP_IN_NCCT_DIR = os.path.join(REG_ROOT, "ctp_in_ncct")
CTP_3D_REF_DIR = os.path.join(REG_ROOT, "CTP_3d_ref_in_ncct")
TRANSFORMS_DIR = os.path.join(REG_ROOT, "transforms")

# Step 3 (optional): CTP parametric maps in NCCT space
PARAM_MAPS_NIFTI_DIR = os.path.join(REG_ROOT, "parametric_maps_nifti")

# Step 4: CTP labels in NCCT space
RAW_LABELS_CTP_DIR = os.path.join(REG_ROOT, "raw_labels_ctp")
LABELS_IN_NCCT_DIR = os.path.join(REG_ROOT, "labels_in_ncct")

# Step 5: DWI in NCCT space
DWI_NIFTI_DIR = os.path.join(REG_ROOT, "DWI/preprocessed")
DWI_IN_NCCT_DIR = os.path.join(REG_ROOT, "DWI/preprocessed")
DWI_TRANSFORMS_DIR = os.path.join(REG_ROOT, "transforms/dwi")

# Step 5: DWI mask + ADC in NCCT space
DWI_MASKS_DIR = os.path.join(REG_ROOT, "DWI/masks")
ADC_RAW_DIR = os.path.join(REG_ROOT, "ADC/raw_data")
ADC_REG_DIR = os.path.join(REG_ROOT, "ADC/registered")

# ─── Skull stripping ────────────────────────────────────────────────
NCCT_SKULL_STRIPPED_DIR = os.path.join(SUS_ROOT, "ncct_skull_stripped")
BRAIN_MASKS_DIR = os.path.join(SUS_ROOT, "stripped_brain_masks")
CTP_SS_DIR = os.path.join(REG_ROOT, "ctp_in_ncct_skullstripped")
DWI_SS_DIR = os.path.join(REG_ROOT, "DWI/preprocessed_skullstripped")
CTP_SS_NATIVE_DIR = os.path.join(REG_ROOT, "ctp_in_ncct_skullstripped_native_mask")
CTP_SS_BRAINMASKS_DIR = os.path.join(REG_ROOT, "ctp_in_ncct_brainmasks")

# ─── Post-registration processing ───────────────────────────────────
FILTERED_LABELS_DIR = os.path.join(REG_ROOT, "filtered_ctp_labels_in_ncct")
SAFE_LABELS_DIR = os.path.join(REG_ROOT, "safe_labels_ctp_in_ncct")
SYNTHSEG_DIR = os.path.join(REG_ROOT, "synthseg_segmentation")
CLB_DIR = os.path.join(REG_ROOT, "CLB_filtered_segmentation")
SIX_CLASS_DIR = os.path.join(REG_ROOT, "6_Class_Labels")
SIX_CLASS_CLEANED_DIR = os.path.join(REG_ROOT, "6_Class_Labels_Cleaned")

# CTP windowing
CTP_WINDOWED_DIR = os.path.join(REG_ROOT, "ctp_in_ncct_windowed_and_skullstripped")

# ─── Curated dataset ────────────────────────────────────────────────
CURATED_ROOT = os.path.join(REG_ROOT, "curated_dataset")

# ─── CLB filtering ──────────────────────────────────────────────────
# JSON mapping patient → lesion side (left/right)
LESION_SIDE_JSON = os.path.join(os.path.dirname(__file__), "post_reg_processing", "lesion.json")

# ─── SynthSeg segmentation labels ───────────────────────────────────
# Left hemisphere labels used for CLB extraction
LEFT_LABELS = [2, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28]
RIGHT_LABELS = [41, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60]
