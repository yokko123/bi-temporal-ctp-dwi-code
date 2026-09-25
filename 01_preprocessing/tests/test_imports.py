#!/usr/bin/env python3
"""Smoke test: verify every module in the pipeline can be imported."""

import importlib
import sys

MODULES = [
    "config",
    # Registration
    "registration.s00_extract_dicom_headers",
    "registration.s01_ctp_motion_correction",
    "registration.s02_ncct_dicom_to_nifti",
    "registration.s03_ctp_to_ncct",
    "registration.s04_dwi_dicom_to_nifti",
    "registration.s05_dwi_adc_to_ncct",
    # Post-registration processing
    "post_reg_processing.p01_filter_ctp_labels",
    "post_reg_processing.p02_safety_margins",
    "post_reg_processing.p03_clb_filtering",
    "post_reg_processing.p04_generate_6_class_labels",
    "post_reg_processing.p05_ctp_windowing",
    "post_reg_processing.p06_clean_ventricle_noise",
    "post_reg_processing.p07_remove_clb_ctp_overlap",
    # Skull stripping
    "skull_stripping.skull_strip_ncct",
    "skull_stripping.skull_strip_ctp_dwi",
    "skull_stripping.skull_strip_ctp_native_masks",
    # Curation
    "curation.copy_ncct",
    "curation.copy_derivatives",
    "curation.copy_6class",
    "curation.copy_raw_ctp",
    # Analysis
    "analysis.voxel_count_2_classes",
    "analysis.voxel_count_6_classes",
    "analysis.voxel_count_dwi",
    "analysis.cc_ctp",
    "analysis.cc_dwi",
    "analysis.collect_image_metadata",
    "analysis.generate_6class_visualizations",
]


def main():
    passed, failed = 0, 0
    for mod_name in MODULES:
        try:
            importlib.import_module(mod_name)
            print(f"  OK   {mod_name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL {mod_name}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"  {passed} passed, {failed} failed out of {len(MODULES)} modules")
    print(f"{'='*50}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
