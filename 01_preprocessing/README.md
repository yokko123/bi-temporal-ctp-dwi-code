# Stage 01 - preprocessing, registration and ROI construction

Turns raw acquisitions into the six outcome-aware ROI classes that everything
downstream is defined on (Fig. 1, top).

Vendored from the project's standalone preprocessing repository,
`yokko123/SUS2025_preprocessing_and_registration_final`, with the institutional
paths replaced by environment variables. Go there for its history and issues.

## Local cohort

```bash
python run_pipeline.py --help          # master CLI; each step also runs standalone
```

| Step | Script | What it does |
|---|---|---|
| s00 | `registration/s00_extract_dicom_headers.py` | DICOM metadata -> CSV |
| s01 | `registration/s01_ctp_motion_correction.py` | CTP DICOM -> 4D NIfTI, rigid motion correction to frame 1, temporal resampling of the last 10 frames to 1 fps (40 frames total) |
| s02 | `registration/s02_ncct_dicom_to_nifti.py` | NCCT DICOM -> NIfTI (the common reference space) |
| s03 | `registration/s03_ctp_to_ncct.py` | 3D rigid CTP -> NCCT; transform propagated to all 40 frames, labels and parametric maps |
| s04 | `registration/s04_dwi_dicom_to_nifti.py` | DWI DICOM -> NIfTI |
| s05 | `registration/s05_dwi_adc_to_ncct.py` | 3D affine DWI + ADC + masks -> NCCT |
| ss  | `skull_stripping/` | SynthStrip on NCCT, then the brain mask applied to CTP and DWI |
| p01 | `post_reg_processing/p01_filter_ctp_labels.py` | drop the brain class from the CTP labels |
| p02 | `post_reg_processing/p02_safety_margins.py` | dilate core, subtract from penumbra; dilate penumbra, subtract from remaining brain |
| p03 | `post_reg_processing/p03_clb_filtering.py` | contralateral hemisphere from SynthSeg |
| **p04** | `post_reg_processing/p04_generate_6_class_labels.py` | **intersect T1 tissue with T2 outcome into the six ROI classes** |
| p05 | `post_reg_processing/p05_ctp_windowing.py` | window CTP to [0, 100] HU, rescale to 8-bit |
| p06 | `post_reg_processing/p06_clean_ventricle_noise.py` | drop labels inside ventricles (SynthSeg) |
| p07 | `post_reg_processing/p07_remove_clb_ctp_overlap.py` | drop CLB voxels overlapping CTP labels |

`curation/` copies the results into the curated layout the feature extractors
read; `analysis/` holds QC (voxel counts, connected components, overlays).

Label codes written by p04, and used by every stage after it:

```
1 core_fi   2 core_brain   3 pen_fi   4 pen_brain   5 clb_brain   6 nhb_fi
```

## ISLES'24

`isles24/` adapts the same construction to ISLES'24, which ships CTP and DWI
already co-registered to NCCT and provides no manual T1 annotation:

* `assemble_isles24_4d.py` - stack the 40 CTP channels into a 4D volume
* `detect_isles_lesion_side.py` - infer the lesion hemisphere
* `generate_6_class_labels_isles.py` - the same six-class intersection, with
  core and penumbra coming from the nnU-Net trained on the local cohort
* `generate_isles_6class_visualizations.py` - QC overlays

## Dependencies

SimpleITK, nibabel, numpy, pandas, tqdm, plus
[SynthStrip](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/) and
[SynthSeg](https://github.com/BBillot/SynthSeg) as external tools.

```bash
pip install -r requirements.txt
```
