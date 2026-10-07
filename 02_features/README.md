# Stage 02 - feature extraction

Four feature families over the preprocessed, windowed CTP, each read out inside
the six ROI masks from stage 01 (Fig. 1, bottom). All four write the same shape
of table: one row per (patient, slice, region), aggregated to patient level
downstream by max-pooling.

| | Family | Dim | Script |
|---|---|---|---|
| FE1 | baseline first-order statistics | 6 | `fe1_baseline/` |
| FE2 | GLCM radiomics | 4 of 24 | `fe2_glcm/` |
| FE3 | mJ-Net encoder embeddings | 256 | `fe3_mjnet/` |
| FE4 | nnU-Net encoder embeddings | 256 | `fe4_nnunet/` |

## FE1 - baseline statistics

Mean, standard deviation, skewness, kurtosis, minimum and maximum over a
3 x 3 x 40 sliding window (stride 1) on the spatiotemporal CTP slice.

```bash
python fe1_baseline/baseline_features_nifti.py       --help   # SUH cohort
python fe1_baseline/baseline_features_nifti_isles.py --help   # ISLES'24
```

CPU only; the paper used a 64-core machine with 64 GB RAM.

## FE2 - GLCM radiomics

Each axial CTP slice is treated as a 3D spatiotemporal volume (two spatial axes
plus time). PyRadiomics extracts 24 symmetrical GLCM features per slice; the
paper retains **Imc1, Imc2, MCC and Correlation**, chosen by Mann-Whitney U with
FDR correction and Cliff's delta.

```bash
python fe2_glcm/3d_glcm.py       --help
python fe2_glcm/3d_glcm_isles.py --help
bash   fe2_glcm/3d_glcm.sh                # SLURM array over patients
```

Needs `pyradiomics` and `SimpleITK`.

## FE3 - mJ-Net embeddings

A 2D+time CNN for core and penumbra segmentation from CTP. Trained on the full
SUH cohort (n=149) with a 116/33 train/test split and 5-fold cross-validation
inside the training set; 100 epochs, AdamW, lr 1e-3, weight decay 1e-2, cosine
annealing, Focal Tversky loss, early stopping patience 20, on A100 80 GB.

```bash
python fe3_mjnet/train.py            --help
python fe3_mjnet/predict.py          --help
python fe3_mjnet/extract_features.py --help   # encoder read-out
```

Feature read-out: 16 x 16 x 40 spatiotemporal patches at stride 1 through the
encoder; the last convolutional block's activations (4 x 4 x 1 x 256) are
global-average-pooled to one 256-D vector per patch centre, then max-pooled over
the voxels of an ROI per slice.

Derived from the reference mJ-Net
([Tomasetti et al.](https://github.com/Biomedical-Data-Analysis-Laboratory/mJ-Net)),
reimplemented in PyTorch; `Architectures/arch_mJNet.py` documents the
layer-by-layer correspondence to the TensorFlow original.

## FE4 - nnU-Net embeddings

A 3D nnU-Net with the 40 CTP time points stacked as input channels, used as a
widely-adopted segmentation backbone for comparison. Self-configuring framework,
500 epochs, lr 5e-3, weight decay 1e-4, 250/50 train/val iterations per epoch,
85% foreground oversampling, deep supervision.

```bash
python fe4_nnunet/extract_3D_ensemble_features.py       --help
python fe4_nnunet/extract_3D_ensemble_features_isles.py --help
```

The volume is tiled into overlapping 3D patches (50% overlap); stage-3 encoder
features (256-D, matching mJ-Net) are extracted from the 5-fold ensemble,
upsampled, stitched into a full-volume map and resampled to NCCT space. Output
carries all three pooling operators as separate column families
(`f0_mean`, `f0_median`, `f0_max`, ...), which is what the Table IV ablation
switches between.

The same SUS-trained model produces the ISLES'24 core and penumbra masks; on a
held-out local test set (n=33) it reached Dice 0.71 +/- 0.24 (penumbra) and
0.30 +/- 0.28 (core).

Needs `nnunetv2`, `torch` and `blosc2`.

## Dependencies

```bash
pip install -r requirements.txt
```

FE1 and FE2 are CPU only. FE3 and FE4 need a GPU plus PyTorch and nnU-Net
respectively, so install only the subset you intend to run; the file is grouped
by feature family.

## Paths

Every script reads its roots from environment variables with `/path/to/...`
defaults; see the table in [../README.md](../README.md#data). Most also accept
the same values as command-line arguments, which take precedence.
