"""
Optuna hyperparameter search for mJNet on 4D CTP data.

Usage:
    python optuna_search.py \
        --derivatives_dir /path/to/curated-dataset \
        --n_trials 30 \
        --num_patients 10

The script maximises the validation *foreground Dice* (mean of brain,
penumbra, core) using Optuna's TPE sampler.  Each trial trains for a
limited number of epochs (configurable via --max_epochs) so the search
finishes in reasonable time; once the best hypers are found you can do a
full training run with train.py.

Results are stored in an SQLite database so you can resume / visualize:
    optuna-dashboard sqlite:///optuna_mjnet.db

Key hyperparameters searched:
    - learning_rate          (log-uniform 1e-4 … 1e-2)
    - weight_decay           (log-uniform 1e-6 … 1e-3)
    - core_class_weight      (uniform 100 … 800)
    - penumbra_class_weight  (uniform 10 … 100)
    - optimizer              (SGD / Adam / AdamW)
    - scheduler              (step / cosine / plateau / none)
    - batch_size             (categorical)
    - stride                 (categorical)
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import numpy as np

try:
    import optuna
    from optuna.trial import Trial
except ImportError:
    print("Optuna not found. Install with:  pip install optuna")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from Data.dataset import get_data_loaders
from Utils.losses import get_loss_function, DiceScore
from Architectures.arch_mJNet import create_mjnet


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("optuna_search")


# ---------------------------------------------------------------------------
# Single-trial training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler, use_amp):
    """Stripped-down training loop (no tqdm/logging overhead)."""
    model.train()
    total_loss = 0.0
    n = 0
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True).unsqueeze(1)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if use_amp and scaler is not None:
            with autocast(device_type="cuda"):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def validate(model, dataloader, criterion, device, num_classes):
    """Return val_loss and per-class Dice."""
    model.eval()
    total_loss = 0.0
    n = 0
    dice_metric = DiceScore(num_classes=num_classes).to(device)
    dice_sum = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True).unsqueeze(1)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            dice_sum += dice_metric(outputs, labels)
            n += 1

    avg_dice = dice_sum / max(n, 1)
    return {
        "val_loss": total_loss / max(n, 1),
        "dice_background": avg_dice[0].item(),
        "dice_brain": avg_dice[1].item(),
        "dice_penumbra": avg_dice[2].item(),
        "dice_core": avg_dice[3].item(),
    }


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def create_objective(args):
    """Factory that captures CLI args and returns the Optuna objective."""

    def objective(trial: Trial) -> float:
        # ---- Suggest hyperparameters ----
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        core_weight = trial.suggest_float("core_weight", 100.0, 800.0)
        penumbra_weight = trial.suggest_float("penumbra_weight", 10.0, 100.0)
        brain_weight = trial.suggest_float("brain_weight", 0.05, 1.0, log=True)
        optimizer_name = trial.suggest_categorical("optimizer", ["sgd", "adam", "adamw"])
        scheduler_name = trial.suggest_categorical("scheduler", ["step", "cosine", "plateau", "none"])
        batch_size = trial.suggest_categorical("batch_size", [1024, 2048, 4096])
        stride = trial.suggest_categorical("stride", [4, 8])

        class_weights = [0.0, brain_weight, penumbra_weight, core_weight]

        logger.info(
            f"\n{'='*60}\n"
            f"Trial {trial.number}  |  lr={lr:.2e}  wd={weight_decay:.2e}\n"
            f"  weights=[0, {brain_weight:.3f}, {penumbra_weight:.1f}, {core_weight:.1f}]\n"
            f"  opt={optimizer_name}  sched={scheduler_name}  bs={batch_size}  stride={stride}\n"
            f"{'='*60}"
        )

        device = torch.device("cuda:0")

        # ---- Config ----
        config = Config()
        config.data.patch_size = args.patch_size
        config.data.stride = stride
        config.data.num_workers = args.num_workers
        config.data.val_split = 0.2
        config.data.class_weights = class_weights
        config.data.cache_dir = args.cache_dir
        config.train.batch_size = batch_size
        config.train.learning_rate = lr
        config.train.weight_decay = weight_decay
        config.train.optimizer = optimizer_name
        config.train.scheduler = scheduler_name
        config.train.use_amp = args.amp
        config.train.augmentation = not args.no_augmentation
        config.train.epochs = args.max_epochs

        # ---- Data ----
        train_loader, val_loader = get_data_loaders(
            config=config,
            derivatives_dir=getattr(args, 'derivatives_dir', None),
            data_dir=getattr(args, 'data_dir', None),
            label_dir=getattr(args, 'label_dir', None),
            num_patients=args.num_patients,
        )
        logger.info(f"  Train patches: {len(train_loader.dataset)}  Val patches: {len(val_loader.dataset)}")

        # ---- Model ----
        model = create_mjnet(
            input_shape=(config.data.timepoints, config.data.patch_size, config.data.patch_size),
            num_classes=config.data.num_classes,
        ).to(device)

        # ---- Loss ----
        criterion = get_loss_function(
            loss_name="dice",
            num_classes=config.data.num_classes,
            class_weights=class_weights,
        )
        if hasattr(criterion, "to"):
            criterion = criterion.to(device)

        # ---- Optimizer ----
        if optimizer_name == "sgd":
            opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                            nesterov=True, weight_decay=weight_decay)
        elif optimizer_name == "adam":
            opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # ---- Scheduler ----
        if scheduler_name == "step":
            sched = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)
        elif scheduler_name == "cosine":
            sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_epochs, eta_min=1e-7)
        elif scheduler_name == "plateau":
            sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5, min_lr=1e-7)
        else:
            sched = None

        scaler = GradScaler("cuda") if args.amp else None

        # ---- Train ----
        best_fg_dice = 0.0
        patience_counter = 0
        patience = 8  # early stopping within each trial

        for epoch in range(args.max_epochs):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, criterion, opt, device, scaler, args.amp)

            # Scheduler step
            if sched is not None:
                if isinstance(sched, optim.lr_scheduler.ReduceLROnPlateau):
                    pass  # step after val
                else:
                    sched.step()

            metrics = validate(model, val_loader, criterion, device, config.data.num_classes)

            if isinstance(sched, optim.lr_scheduler.ReduceLROnPlateau):
                sched.step(metrics["val_loss"])

            fg_dice = (metrics["dice_brain"] + metrics["dice_penumbra"] + metrics["dice_core"]) / 3.0
            elapsed = time.time() - t0

            logger.info(
                f"  Epoch {epoch+1}/{args.max_epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={metrics['val_loss']:.4f}  "
                f"FG Dice={fg_dice:.4f}  (brain={metrics['dice_brain']:.3f} "
                f"pen={metrics['dice_penumbra']:.3f} core={metrics['dice_core']:.3f})  "
                f"[{elapsed:.1f}s]"
            )

            if fg_dice > best_fg_dice:
                best_fg_dice = fg_dice
                patience_counter = 0
            else:
                patience_counter += 1

            # Report intermediate value so Optuna can prune bad trials
            trial.report(fg_dice, epoch)
            if trial.should_prune():
                logger.info(f"  Trial {trial.number} PRUNED at epoch {epoch+1}")
                raise optuna.exceptions.TrialPruned()

            if patience_counter >= patience:
                logger.info(f"  Early stopped at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

        logger.info(f"  Trial {trial.number} finished — best FG Dice = {best_fg_dice:.4f}")

        # Clean up GPU memory
        del model, opt, criterion, scaler, train_loader, val_loader
        torch.cuda.empty_cache()

        return best_fg_dice

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for mJNet")

    # Data
    parser.add_argument("--derivatives_dir", type=str,
                        default=os.environ.get("SUS_CURATED_ROOT", "/path/to/Curated-SUS-2025"),
                        help="Curated dataset root (contains derivatives/ folder)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="(Legacy) CTP directory — overrides --derivatives_dir with --label_dir")
    parser.add_argument("--label_dir", type=str, default=None,
                        help="(Legacy) Label directory — overrides --derivatives_dir with --data_dir")
    parser.add_argument("--cache_dir", type=str, default="Data/npz_cache")
    parser.add_argument("--num_patients", type=int, default=None,
                        help="Limit patients for faster search")

    # Search budget
    parser.add_argument("--n_trials", type=int, default=30,
                        help="Number of Optuna trials")
    parser.add_argument("--max_epochs", type=int, default=15,
                        help="Max epochs per trial (keep small for speed)")

    # Fixed settings
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--amp", action="store_true", help="Mixed precision")
    parser.add_argument("--no-augmentation", action="store_true")

    # Optuna settings
    parser.add_argument("--study_name", type=str, default="mjnet_hparam_search")
    parser.add_argument("--db", type=str, default="sqlite:///optuna_mjnet.db",
                        help="Optuna storage (SQLite). Allows resume & dashboard.")
    parser.add_argument("--pruner", type=str, default="median",
                        choices=["median", "hyperband", "none"],
                        help="Pruning strategy to stop bad trials early")

    args = parser.parse_args()

    # ---- Pruner ----
    if args.pruner == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,        # Don't prune first 5 trials
            n_warmup_steps=3,          # Don't prune before epoch 3
            interval_steps=1,
        )
    elif args.pruner == "hyperband":
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=3,
            max_resource=args.max_epochs,
            reduction_factor=3,
        )
    else:
        pruner = optuna.pruners.NopPruner()

    # ---- Study ----
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.db,
        direction="maximize",          # Maximise foreground Dice
        pruner=pruner,
        load_if_exists=True,           # Resume if DB already exists
    )

    logger.info(f"Study '{args.study_name}' — {args.n_trials} trials, {args.max_epochs} epochs/trial")
    logger.info(f"Database: {args.db}")
    logger.info(f"Existing trials: {len(study.trials)}")

    # ---- Run ----
    study.optimize(
        create_objective(args),
        n_trials=args.n_trials,
        timeout=None,
        show_progress_bar=True,
    )

    # ---- Report ----
    logger.info("\n" + "=" * 70)
    logger.info("OPTUNA SEARCH COMPLETED")
    logger.info("=" * 70)

    logger.info(f"\nBest trial: #{study.best_trial.number}")
    logger.info(f"Best foreground Dice: {study.best_value:.4f}")
    logger.info(f"Best hyperparameters:")
    for k, v in study.best_params.items():
        logger.info(f"  {k}: {v}")

    # ---- Save summary ----
    summary_path = "optuna_results.txt"
    with open(summary_path, "w") as f:
        f.write(f"Study: {args.study_name}\n")
        f.write(f"Best trial: #{study.best_trial.number}\n")
        f.write(f"Best foreground Dice: {study.best_value:.4f}\n\n")
        f.write("Best hyperparameters:\n")
        for k, v in study.best_params.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nTotal trials: {len(study.trials)}\n")
        f.write(f"  Completed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}\n")
        f.write(f"  Pruned:    {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}\n")
        f.write(f"  Failed:    {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}\n")

        f.write("\n\nAll completed trials (sorted by value):\n")
        f.write(f"{'#':>4s}  {'FG Dice':>8s}  {'LR':>10s}  {'WD':>10s}  {'Core W':>7s}  {'Pen W':>6s}  {'Opt':>5s}  {'Sched':>7s}  {'BS':>5s}  {'Stride':>6s}\n")
        f.write("-" * 90 + "\n")
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        for t in sorted(completed, key=lambda t: t.value, reverse=True):
            p = t.params
            f.write(
                f"{t.number:4d}  {t.value:8.4f}  {p['lr']:10.2e}  {p['weight_decay']:10.2e}  "
                f"{p['core_weight']:7.1f}  {p['penumbra_weight']:6.1f}  {p['optimizer']:>5s}  "
                f"{p['scheduler']:>7s}  {p['batch_size']:5d}  {p['stride']:6d}\n"
            )

    logger.info(f"\nResults saved to: {summary_path}")

    # ---- Print command for full training with best params ----
    bp = study.best_params
    logger.info("\n" + "=" * 70)
    logger.info("To train with the best hyperparameters, run:")
    logger.info("=" * 70)

    # Build class_weights string for reference
    cw = f"[0.0, {bp['brain_weight']:.3f}, {bp['penumbra_weight']:.1f}, {bp['core_weight']:.1f}]"
    # Build data path arg for the suggested command
    if args.data_dir and args.label_dir:
        data_args = (
            f"  --data_dir \"{args.data_dir}\" \\\n"
            f"  --label_dir \"{args.label_dir}\" \\\n"
        )
    else:
        data_args = f"  --derivatives_dir \"{args.derivatives_dir}\" \\\n"
    logger.info(
        f"\npython train.py \\\n"
        f"{data_args}"
        f"  --epochs 40 \\\n"
        f"  --batch_size {bp['batch_size']} \\\n"
        f"  --lr {bp['lr']:.6f} \\\n"
        f"  --optimizer {bp['optimizer']} \\\n"
        f"  --scheduler {bp['scheduler']} \\\n"
        f"  --patch_size {args.patch_size} \\\n"
        f"  --stride {bp['stride']} \\\n"
        f"  --num_workers {args.num_workers} \\\n"
        f"  --amp \\\n"
        f"  --num_patients {args.num_patients or 'ALL'}\n"
    )
    logger.info(f"NOTE: Update class_weights in config.py to: {cw}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
