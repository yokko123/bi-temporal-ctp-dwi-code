#!/bin/bash
#SBATCH --job-name=extract_feat
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/extract_feat_%A_%a.out
#SBATCH --error=logs/extract_feat_%A_%a.err
#SBATCH --array=0-4

# ---- Usage ----
# Single job (all patients):
#   sbatch --array=0 extract_features.sh
#
# Chunked (5 chunks in parallel):
#   sbatch extract_features.sh
#
# Adjust --array=0-N to match CHUNKS below.
# After all chunks complete, concatenate CSVs (skip headers for chunks 1+):
#   head -n 1 encoder_features_6class_stride1_chunks5_idx0.csv > encoder_features_6class_stride1_all.csv
#   for i in $(seq 0 4); do tail -n +2 encoder_features_6class_stride1_chunks5_idx${i}.csv >> encoder_features_6class_stride1_all.csv; done
# -------------------

CHUNKS=5   # must match --array range (0 to CHUNKS-1)
STRIDE=1

# Create log directory
mkdir -p logs

# Activate conda environment
# source ${CONDA_PREFIX_ROOT:-/path/to/miniconda3}/etc/profile.d/conda.sh
# conda activate test_tensorflow_env_py310

cd ${USER_WORK_ROOT:-/path/to/work}/"PhD works"/sus25_feat_extraction/mJ-Net_pytorch

python extract_features_isles.py \
    --stride ${STRIDE} \
    --batch_size 2048 \
    --chunks ${CHUNKS} \
    --chunk_idx ${SLURM_ARRAY_TASK_ID} \
    --allow_unknown_train
