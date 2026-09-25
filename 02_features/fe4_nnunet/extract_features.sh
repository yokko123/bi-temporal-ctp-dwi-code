#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH -w gorina9
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=extract_3d_feats
#SBATCH --output=extract_3d_feats_%j.out
#SBATCH --error=extract_3d_feats_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Job started on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

python3 extract_3D_ensemble_features_isles.py --device cuda --resolution ncct

echo "Job finished at $(date)"
