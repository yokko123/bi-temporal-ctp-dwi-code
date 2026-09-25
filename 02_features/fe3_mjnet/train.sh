#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=5-00:00:00
#SBATCH --job-name=mJNet_train
#SBATCH --output=mjnet_pytorch7.out
#SBATCH --error=mjnet_pytorch7.err

set -euo pipefail

# Navigate to project directory
# cd "${FEATURE_WORK_ROOT:-/path/to/feature-work}/mJ-Net_pytorch"

# Activate conda environment
# source "${CONDA_PREFIX_ROOT:-/path/to/miniconda3}/bin/activate" base
# Print environment info
echo "=========================================="
echo "Starting mJNet Training"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=========================================="

# Run training
# Loss: focal_tversky (alpha=0.2, beta=0.8, gamma=1.5) + label smoothing 0.02
#   - beta>alpha penalises false negatives (missed lesion) more than FPs
#   - Higher beta/gamma vs. previous run to push harder on rare classes
#   - Background excluded from loss (class_weights 0 1 1 1)
# Optimiser: AdamW with decoupled weight decay 1e-2
# Scheduler: cosine annealing over 100 epochs (smooth LR decay, no plateau lag)
# Augmentation: fully stochastic (random geometric + intensity per sample)
# Speed: vectorised Gaussian blur (15× faster), stochastic aug (no index expansion)
#
# Key anti-overfitting measures:
#   - Dropout3d rate=0.2 at all designated locations
#   - AdamW weight_decay=1e-2 (decoupled, stronger than Adam+L2)
#   - Label smoothing eps=0.02 in loss
#   - Stochastic augmentation (each sample gets random transform, not 6× fixed)
#   - Gradient clipping max_norm=1.0 (built-in)
#
# Changes vs. previous runs:
#   - class_weights 0 1 1 1: excludes background from loss → focuses gradients on fg
#   - oversample_lesion 6 (was 2): more lesion patches per epoch
#   - lr 1e-3 (was 3e-4): higher LR, cosine scheduler will anneal
#   - stride 8: best dice and fastest (stride 4/2 were slower with worse dice)
#   - FocalTversky alpha=0.2/beta=0.8/gamma=1.5 (was 0.3/0.7/1.33)
#   - label_smoothing 0.02 (was 0.05): less aggressive for rare classes

python train.py \
  --derivatives_dir "${SUS_CURATED_ROOT:-/path/to/Curated-SUS-2025}" \
  --epochs 100 \
  --batch_size 2048 \
  --num_workers 8 \
  --amp \
  --patch_size 16 \
  --stride 8 \
  --loss focal_tversky \
  --class_weights 0 1 1 1 \
  --optimizer adamw \
  --scheduler cosine \
  --lr 1e-3 \
  --weight_decay 1e-2 \
  --dropout \
  --patience 20 \
  --oversample_lesion 6 \
  --num_folds 5 \
  --steps_per_epoch 5000 \
  --test_patients \
    01_001 01_007 01_013 01_019 01_025 01_031 01_037 01_044 01_049 01_053 \
    01_061 01_067 01_074 02_001 02_007 02_013 02_019 02_025 02_031 02_036 \
    02_043 02_050 02_055 02_062 03_003 03_010 03_014 01_057 01_059 01_066 01_068 01_071 01_073 01_032 \
    
echo "=========================================="
echo "Training completed: $(date)"
echo "=========================================="