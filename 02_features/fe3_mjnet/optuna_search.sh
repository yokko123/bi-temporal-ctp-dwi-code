#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=3-00:00:00
#SBATCH --job-name=mJNet_optuna
#SBATCH --output=optuna_search.out
#SBATCH --error=optuna_search.err

set -euo pipefail

echo "=========================================="
echo "Optuna Hyperparameter Search — mJNet"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=========================================="

# Run Optuna search
# - 30 trials, 15 epochs each (enough to see trends, ~30min/trial)
# - Uses 10 patients for speed
# - Results stored in SQLite DB (resumable!)
# - MedianPruner kills bad trials early
python optuna_search.py \
  --derivatives_dir "${SUS_CURATED_ROOT:-/path/to/Curated-SUS-2025}" \
  --n_trials 30 \
  --max_epochs 15 \
  --num_patients 5 \
  --patch_size 16 \
  --num_workers 8 \
  --amp \
  --study_name mjnet_hparam_search \
  --db sqlite:///optuna_mjnet.db \
  --no-augmentation \
  --pruner median
echo "=========================================="
echo "Optuna search completed: $(date)"
echo "Results: optuna_results.txt"
echo "DB:      optuna_mjnet.db"
echo "=========================================="
