#!/bin/bash
# ==========================================================================
# Experiment Launcher — submits multiple SLURM jobs with different configs
#
# Usage:
#   bash launch_experiments.sh          # submit all experiments
#   bash launch_experiments.sh --dry    # preview commands without submitting
# ==========================================================================

set -euo pipefail

PROJECT_DIR="${FEATURE_WORK_ROOT:-/path/to/feature-work}/mJ-Net_pytorch"
DRY_RUN=false
[[ "${1:-}" == "--dry" ]] && DRY_RUN=true

# ── Log directory (MUST be a space-free path — SLURM #SBATCH directives cannot handle spaces) ──
LOG_DIR="${MJNET_LOG_DIR:-/path/to/mjnet_logs}"

# ── Common settings (shared across all experiments) ──────────────────────
DERIVATIVES_DIR="${SUS_CURATED_ROOT:-/path/to/Curated-SUS-2025}"

# ── Define experiments ───────────────────────────────────────────────────
# Each experiment: NAME | LOSS | OPTIMIZER | LR | SCHEDULER | OVERSAMPLE | EXTRA_ARGS
#
# Add/remove/modify rows below to define your experiments.
# Use "--no-augmentation" in EXTRA_ARGS to disable augmentation, or leave "" for default.

EXPERIMENTS=(
    # NAME                    | LOSS           | OPTIMIZER | LR    | SCHEDULER | OVERSAMPLE | EXTRA_ARGS
    "focal_tversky_sgd_3e4    | focal_tversky  | sgd       | 3e-4  | plateau   | 15         | --no-augmentation"
    "focal_tversky_sgd_1e3    | focal_tversky  | sgd       | 1e-3  | plateau   | 15         | --no-augmentation"
    "focal_tversky_adam_1e4   | focal_tversky  | adam      | 1e-4  | plateau   | 15         | --no-augmentation"
    "dice_adam_1e4            | dice           | adam      | 1e-4  | plateau   | 10         | --no-augmentation"
    "squared_dice_sgd_3e4    | squared_dice   | sgd       | 3e-4  | plateau   | 15         | --no-augmentation"
    "combined_adam_1e4        | combined       | adam      | 1e-4  | plateau   | 15         | --no-augmentation"
)

# ── Submit loop ──────────────────────────────────────────────────────────
echo "============================================="
echo "  Experiment Launcher"
echo "  Date: $(date)"
echo "  Dry run: $DRY_RUN"
echo "============================================="

SUBMITTED=0
for EXP in "${EXPERIMENTS[@]}"; do
    # Parse the pipe-separated fields
    IFS='|' read -r NAME LOSS OPTIMIZER LR SCHEDULER OVERSAMPLE EXTRA <<< "$EXP"

    # Trim whitespace
    NAME=$(echo "$NAME" | xargs)
    LOSS=$(echo "$LOSS" | xargs)
    OPTIMIZER=$(echo "$OPTIMIZER" | xargs)
    LR=$(echo "$LR" | xargs)
    SCHEDULER=$(echo "$SCHEDULER" | xargs)
    OVERSAMPLE=$(echo "$OVERSAMPLE" | xargs)
    EXTRA=$(echo "$EXTRA" | xargs)

    JOB_NAME="mJNet_${NAME}"
    SCRIPT_FILE="${LOG_DIR}/${NAME}.slurm"

    echo ""
    echo "── Experiment: ${NAME} ──"
    echo "   Loss=${LOSS}  Opt=${OPTIMIZER}  LR=${LR}  Sched=${SCHEDULER}  Oversample=${OVERSAMPLE}  Extra=${EXTRA}"

    # Create logs directory if needed
    mkdir -p "$LOG_DIR"

    # Write a per-experiment SLURM script
    # NOTE: #SBATCH lines use LOG_DIR (space-free path). Script body quotes paths with spaces.
    cat > "$SCRIPT_FILE" <<SLURM_EOF
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=5-00:00:00
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_DIR}/${NAME}.out
#SBATCH --error=${LOG_DIR}/${NAME}.err

set -euo pipefail
echo "=========================================="
echo "Experiment: ${NAME}"
echo "Date: \$(date)"
echo "Node: \$(hostname)"
echo "GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=========================================="

cd "${PROJECT_DIR}"
python train.py \\
    --derivatives_dir "${DERIVATIVES_DIR}" \\
    --epochs 50 \\
    --batch_size 2048 \\
    --num_workers 8 \\
    --amp \\
    --patch_size 16 \\
    --stride 4 \\
    --loss ${LOSS} \\
    --optimizer ${OPTIMIZER} \\
    --lr ${LR} \\
    --scheduler ${SCHEDULER} \\
    --oversample_lesion ${OVERSAMPLE} \\
    ${EXTRA}

echo "=========================================="
echo "Done: \$(date)"
echo "=========================================="
SLURM_EOF

    if $DRY_RUN; then
        echo "   [DRY RUN] Generated: $SCRIPT_FILE"
        echo "   Would run: sbatch \"$SCRIPT_FILE\""
    else
        sbatch "$SCRIPT_FILE"
        SUBMITTED=$((SUBMITTED + 1))
        echo "   ✓ Submitted (script: $SCRIPT_FILE)"
    fi
done

echo ""
echo "============================================="
if $DRY_RUN; then
    echo "  Dry run complete. ${#EXPERIMENTS[@]} experiments previewed."
    echo "  Run without --dry to submit."
else
    echo "  Done! Submitted ${SUBMITTED} jobs."
    echo "  Check status:  squeue -u \$USER"
    echo "  Logs in:       ${PROJECT_DIR}/logs/"
fi
echo "============================================="
