#!/bin/bash

#SBATCH --job-name=mjnet_predict
#SBATCH --output=slurm_logs/predict_%j.out
#SBATCH --error=slurm_logs/predict_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=0-08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Create log directory
mkdir -p slurm_logs

source ${CONDA_PREFIX_ROOT:-/path/to/miniconda3}/bin/activate test_tensorflow_env_py310

cd "${FEATURE_WORK_ROOT:-/path/to/feature-work}/mJ-Net_pytorch"

# ============================================================
# Choice 1: Predict ALL patients (5-fold ensemble)
# ============================================================
python predict.py \
    --cv_dir checkpoints/cv_5fold_20260310_174037 \
    --derivatives_dir "${SUS_CURATED_ROOT:-/path/to/Curated-SUS-2025}" \
    --output_dir predictions/ensemble_all \
    --stride 4 \
    --batch_size 2048

# ============================================================
# Choice 2: Predict specific patients only (uncomment to use)
# ============================================================
# python predict.py \
#     --cv_dir checkpoints/cv_5fold_20260310_174037 \
#     --derivatives_dir "${SUS_CURATED_ROOT:-/path/to/Curated-SUS-2025}" \
#     --output_dir predictions/ensemble_selected \
#     --patients sub-stroke_01_001 sub-stroke_01_002 sub-stroke_01_026 \
#     --stride 4 \
#     --batch_size 2048

# ============================================================
# Choice 3: Single-fold prediction (uncomment to use)
# ============================================================
# python predict.py \
#     --cv_dir checkpoints/cv_5fold_20260310_174037 \
#     --folds 0 \
#     --derivatives_dir "${SUS_CURATED_ROOT:-/path/to/Curated-SUS-2025}" \
#     --output_dir predictions/fold0_only \
#     --stride 4 \
#     --batch_size 2048
