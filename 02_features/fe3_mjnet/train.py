"""
Training script for mJNet on 4D CTP data.
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
from torch.amp import GradScaler, autocast  # Updated API (torch >= 2.0)
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config, TrainConfig
from Data.dataset import get_data_loaders, get_cv_fold_loaders, CTPVolumeDataset
from Utils.losses import get_loss_function, DiceScore
from Utils.plotting import plot_training_curves, save_volume_predictions
from Architectures.arch_mJNet import create_mjnet


def setup_logging(log_dir: str) -> logging.Logger:
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[object],
    epoch: int,
    best_metric: float,
    checkpoint_path: str,
    is_best: bool = False,
):
    """Save model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        best_path = checkpoint_path.replace(".pt", "_best.pt")
        torch.save(checkpoint, best_path)


def load_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[object],
    checkpoint_path: str,
    device: torch.device,
) -> Dict:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    use_amp: bool = False,
    logger: Optional[logging.Logger] = None,
    steps_per_epoch: int = 0,
) -> Dict[str, float]:
    """Train for one epoch (optionally capped at steps_per_epoch batches)."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    total_steps = steps_per_epoch if steps_per_epoch > 0 else len(dataloader)
    pbar = tqdm(dataloader, desc="Training", leave=True, ncols=100, total=total_steps)
    for batch_idx, (images, labels) in enumerate(pbar):
        if steps_per_epoch > 0 and batch_idx >= steps_per_epoch:
            break
        images = images.to(device)  # (B, T, H, W) -> will reshape in forward
        labels = labels.to(device)  # (B, H, W)
        
        # Add channel dimension: (B, T, H, W) -> (B, 1, T, H, W)
        images = images.unsqueeze(1)
        
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with autocast(device_type='cuda'):
                outputs = model(images)  # (B, C, H, W)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg": f"{total_loss/num_batches:.4f}"})
        
        # Log every 10 batches
        if logger and (batch_idx + 1) % 10 == 0:
            logger.info(f"  Batch [{batch_idx + 1}/{len(dataloader)}] Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / max(num_batches, 1)
    if logger:
        logger.info(f"  Train Loss: {avg_loss:.4f}")
    return {"loss": avg_loss}


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 4,
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    dice_metric = DiceScore(num_classes=num_classes).to(device)
    dice_scores = torch.zeros(num_classes, device=device)
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation", leave=True, ncols=100)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Add channel dimension
            images = images.unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Compute Dice scores
            dice = dice_metric(outputs, labels)
            dice_scores += dice
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_loss = total_loss / max(num_batches, 1)
    avg_dice = dice_scores / max(num_batches, 1)
    
    # Paper: "We excluded background predictions from the calculation of the statistical results"
    foreground_dice = avg_dice[1:].mean().item()  # mean of brain, penumbra, core only
    
    metrics = {
        "val_loss": avg_loss,
        "dice_background": avg_dice[0].item(),
        "dice_brain": avg_dice[1].item(),
        "dice_penumbra": avg_dice[2].item(),
        "dice_core": avg_dice[3].item(),
        "dice_mean": foreground_dice,  # Foreground-only (excludes background)
    }
    
    return metrics


def _run_training(config: Config, args: argparse.Namespace,
                   train_loader, val_loader,
                   run_dir: str, logger) -> Dict:
    """Core training loop — model creation, optimisation, checkpointing.

    Returns a dict with per-epoch history and best metrics.
    """
    device = torch.device("cuda:0")

    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    if config.data.oversample_lesion > 1:
        logger.info(f"Lesion oversampling: {config.data.oversample_lesion}× (patches with penumbra/core repeated)")

    # Create model
    logger.info(f"Creating model: mJNet (longJ={config.train.use_longJ}, v2={config.train.use_v2})")
    model = create_mjnet(
        input_shape=(config.data.timepoints, config.data.patch_size, config.data.patch_size),
        num_classes=config.data.num_classes,
        longJ=config.train.use_longJ,
        v2=config.train.use_v2,
        dropout=config.train.dropout,
        dropout_rates=config.train.dropout_rates if config.train.dropout else None,
    )
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")
    
    # Print model summary with per-layer output shapes
    logger.info("\n" + "=" * 80)
    logger.info("MODEL SUMMARY")
    logger.info("=" * 80)
    input_size = (config.data.timepoints, config.data.patch_size, config.data.patch_size)
    logger.info(f"Input shape: (batch, {input_size[0]}, {input_size[1]}, {input_size[2]})")
    logger.info("-" * 80)
    logger.info(f"{'Layer':<30s} {'Output Shape':<25s} {'Params':>12s}")
    logger.info("-" * 80)
    
    # Use forward hooks to capture output shapes
    layer_shapes = {}
    hooks = []
    
    def get_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                layer_shapes[name] = tuple(output.shape)
            elif isinstance(output, (tuple, list)) and len(output) > 0:
                layer_shapes[name] = tuple(output[0].shape)
        return hook
    
    for name, module in model.named_children():
        hooks.append(module.register_forward_hook(get_hook(name)))
    
    # Run a dummy forward pass to get shapes
    with torch.no_grad():
        # Model expects 5D: (batch, channels=1, timepoints, height, width)
        dummy_input = torch.zeros(1, 1, input_size[0], input_size[1], input_size[2], device=device)
        _ = model(dummy_input)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Print layer info with shapes
    for name, module in model.named_children():
        num_module_params = sum(p.numel() for p in module.parameters())
        shape_str = str(layer_shapes.get(name, "N/A"))
        logger.info(f"{name:<30s} {shape_str:<25s} {num_module_params:>12,}")
    
    logger.info("-" * 80)
    logger.info(f"{'Total':<30s} {'':<25s} {num_params:>12,}")
    logger.info(f"Output shape: (batch, {config.data.num_classes}, {config.data.patch_size}, {config.data.patch_size})")
    logger.info("=" * 80 + "\n")
    
    # Loss function
    logger.info(f"Loss function: {config.train.loss_function}")
    logger.info(f"Class weights: {config.data.class_weights}")
    criterion = get_loss_function(
        loss_name=config.train.loss_function,
        num_classes=config.data.num_classes,
        class_weights=config.data.class_weights,
    )
    logger.info(f"Loss class: {criterion.__class__.__name__}")
    if hasattr(criterion, 'to'):
        criterion = criterion.to(device)
    
    # Optimizer
    if config.train.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
    elif config.train.optimizer == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
    elif config.train.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.train.learning_rate,
            momentum=0.9,
            nesterov=True,  # Matching reference TF code
            weight_decay=config.train.weight_decay,
        )
        logger.info("Using SGD with Nesterov momentum=0.9")
    else:
        raise ValueError(f"Unknown optimizer: {config.train.optimizer}")
    
    # Learning rate scheduler
    if config.train.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.train.epochs,
            eta_min=1e-6,
        )
    elif config.train.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=config.train.lr_patience,
            min_lr=1e-7,
        )
    elif config.train.scheduler == "step":
        # Matching reference: decay every 10 epochs by 0.1
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.1,
        )
        logger.info("Using StepLR scheduler: decay by 0.1 every 10 epochs")
    else:
        scheduler = None
    
    # Mixed precision training
    scaler = GradScaler('cuda') if config.train.use_amp and device.type == "cuda" else None
    if config.train.use_amp:
        logger.info("Mixed precision training (AMP) enabled")
    
    # Resume from checkpoint
    start_epoch = 0
    best_metric = float("inf")
    
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = load_checkpoint(model, optimizer, scheduler, args.resume, device)
        start_epoch = checkpoint["epoch"] + 1
        best_metric = checkpoint["best_metric"]
    
    # Training loop
    logger.info("Starting training...")
    patience_counter = 0

    # ---- Metrics history for plotting ----
    history = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
        "dice_background": [],
        "dice_brain": [],
        "dice_penumbra": [],
        "dice_core": [],
        "dice_mean": [],
        "foreground_dice": [],
    }
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    for epoch in range(start_epoch, config.train.epochs):
        epoch_start = time.time()
        logger.info(f"\nEpoch {epoch + 1}/{config.train.epochs}")
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        sys.stdout.flush()
        
        # Train
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=config.train.use_amp,
            logger=logger,
            steps_per_epoch=getattr(config.train, 'steps_per_epoch', 0),
        )
        
        # Validate
        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=config.data.num_classes,
        )
        
        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["val_loss"])
            else:
                scheduler.step()
        
        # Log metrics
        epoch_time = time.time() - epoch_start
        
        # dice_mean already excludes background (paper: foreground-only stats)
        foreground_dice = val_metrics['dice_mean']
        
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
        logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}")
        logger.info(f"Dice - Background: {val_metrics['dice_background']:.4f}, "
                   f"Brain: {val_metrics['dice_brain']:.4f}, "
                   f"Penumbra: {val_metrics['dice_penumbra']:.4f}, "
                   f"Core: {val_metrics['dice_core']:.4f}")
        logger.info(f"Foreground Dice (brain+pen+core): {foreground_dice:.4f}")
        logger.info(f"Epoch time: {epoch_time:.1f}s")
        sys.stdout.flush()

        # ---- Record history ----
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["val_loss"])
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["dice_background"].append(val_metrics["dice_background"])
        history["dice_brain"].append(val_metrics["dice_brain"])
        history["dice_penumbra"].append(val_metrics["dice_penumbra"])
        history["dice_core"].append(val_metrics["dice_core"])
        history["dice_mean"].append(val_metrics["dice_mean"])
        history["foreground_dice"].append(foreground_dice)

        # ---- Save training curves every epoch (overwrite) ----
        try:
            plot_training_curves(history, plots_dir, filename="training_curves.png")
        except Exception as e:
            logger.warning(f"Could not save training curves: {e}")
        
        # Save checkpoint: only best and last
        is_best = val_metrics["val_loss"] < best_metric
        if is_best:
            best_metric = val_metrics["val_loss"]
            patience_counter = 0
            # Save best checkpoint
            best_path = os.path.join(run_dir, "checkpoint_best.pt")
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_metric,
                checkpoint_path=best_path,
                is_best=False,  # Already saving as best
            )
            logger.info(f"  -> Saved new best checkpoint (val_loss={best_metric:.4f})")
        else:
            patience_counter += 1
        
        # Always save last checkpoint (overwrite)
        last_path = os.path.join(run_dir, "checkpoint_last.pt")
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_metric=best_metric,
            checkpoint_path=last_path,
            is_best=False,
        )
        
        # Early stopping
        if config.train.early_stopping and patience_counter >= config.train.patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_metric:.4f}")

    # ---- Final training curves ----
    try:
        curve_path = plot_training_curves(history, plots_dir, filename="training_curves.png")
        logger.info(f"Training curves saved to: {curve_path}")
    except Exception as e:
        logger.warning(f"Could not save final training curves: {e}")

    # ---- Save prediction PNGs ----
    try:
        # Load best checkpoint for predictions
        best_ckpt = os.path.join(run_dir, "checkpoint_best.pt")
        if os.path.exists(best_ckpt):
            logger.info("Loading best checkpoint for prediction visualisations...")
            ckpt = torch.load(best_ckpt, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])

        # Build a volume-level dataset from validation data
        # Reuse the original NIfTI paths stored on the patch dataset
        val_ds = val_loader.dataset
        val_ctp_paths = getattr(val_ds, 'ctp_paths', [])
        val_label_paths = getattr(val_ds, 'label_paths', [])

        if val_ctp_paths:
            from torch.utils.data import DataLoader as DL
            vol_dataset = CTPVolumeDataset(
                ctp_paths=val_ctp_paths,
                label_paths=val_label_paths if val_label_paths else None,
                cache_dir=config.data.cache_dir,
                patch_size=config.data.patch_size,
                stride=config.data.stride,
            )
            vol_loader = DL(vol_dataset, batch_size=1, shuffle=False, num_workers=0)

            pred_dir = os.path.join(run_dir, "predictions")
            logger.info(f"Saving prediction PNGs to {pred_dir}...")
            logger.info(f"Found {len(val_ctp_paths)} validation CTP files")
            logger.info(f"Volume dataset has {len(vol_dataset)} slices")
            
            n_saved = save_volume_predictions(
                model=model,
                volume_dataset=vol_dataset,
                device=device,
                save_root=pred_dir,
                max_volumes=len(val_ctp_paths),  # Process ALL validation patients, not just 5
                skip_empty=True,
                top_k_slices=5,
                logger=logger,
            )
            logger.info(f"Successfully saved predictions for {n_saved} volumes to {pred_dir}")
        else:
            logger.warning("Could not find validation NIfTI paths for prediction saving.")
    except Exception as e:
        logger.warning(f"Could not save predictions: {e}")
        import traceback
        logger.warning(traceback.format_exc())

    # Return summary for cross-validation aggregation
    return {
        "history": history,
        "best_val_loss": best_metric,
        "best_foreground_dice": max(history["foreground_dice"]) if history["foreground_dice"] else 0.0,
        "best_dice_penumbra": max(history["dice_penumbra"]) if history["dice_penumbra"] else 0.0,
        "best_dice_core": max(history["dice_core"]) if history["dice_core"] else 0.0,
    }


# ======================================================================
# Entry points: single split  vs.  k-fold cross-validation
# ======================================================================

def train(config: Config, args: argparse.Namespace):
    """Single train/val split training (original workflow)."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! This script requires GPU.")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config.train.checkpoint_dir, f"run_{run_timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    logger = setup_logging(os.path.join(run_dir, "logs"))
    logger.info("=" * 60)
    logger.info(f"Run directory: {run_dir}")
    logger.info("=" * 60)
    logger.info(f"Using device: cuda:0")
    logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    logger.info("Loading data...")
    test_patients = getattr(config.data, 'test_patients', None)
    if test_patients:
        logger.info(f"Holding out test patients: {test_patients}")

    train_loader, val_loader = get_data_loaders(
        config=config,
        derivatives_dir=getattr(args, 'derivatives_dir', None),
        data_dir=getattr(args, 'data_dir', None),
        label_dir=getattr(args, 'label_dir', None),
        num_patients=getattr(args, 'num_patients', None),
        test_patients=test_patients,
    )

    _run_training(config, args, train_loader, val_loader, run_dir, logger)


# ======================================================================
#  Held-out test set evaluation (called after all CV folds)
# ======================================================================

def _evaluate_test_set(
    config: Config,
    args: argparse.Namespace,
    cv_dir: str,
    test_patients: list,
    num_folds: int,
    logger,
):
    """Load all fold best-checkpoints, ensemble-predict on held-out test
    patients, save NIfTI + PNG predictions with GT contours, and report
    per-patient / aggregate Dice metrics.
    """
    import re
    import SimpleITK as sitk
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dir = os.path.join(cv_dir, "test_predictions")
    os.makedirs(test_dir, exist_ok=True)

    # ---- Load fold models ----
    models = []
    for fold_idx in range(num_folds):
        ckpt_path = os.path.join(cv_dir, f"fold_{fold_idx}", "checkpoint_best.pt")
        if not os.path.exists(ckpt_path):
            logger.warning(f"  Fold {fold_idx}: checkpoint_best.pt not found, skipping")
            continue
        model = create_mjnet(
            input_shape=(config.data.timepoints, config.data.patch_size, config.data.patch_size),
            num_classes=config.data.num_classes,
            dropout=False,
        )
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device).eval()
        models.append(model)
        logger.info(f"  Loaded fold_{fold_idx}/checkpoint_best.pt (epoch {ckpt.get('epoch', '?')})")

    if not models:
        logger.error("  No fold checkpoints found — cannot evaluate test set.")
        return

    logger.info(f"  Ensemble of {len(models)} fold models")

    # ---- Discover test patient files ----
    derivatives_dir = getattr(args, 'derivatives_dir', None)
    if derivatives_dir is None:
        logger.error("  No derivatives_dir — cannot discover test files.")
        return

    test_files = []  # list of (patient_id, ctp_path, label_path)
    deriv_root = os.path.join(derivatives_dir, "derivatives")
    for subj in sorted(os.listdir(deriv_root)):
        if not re.match(r"sub-stroke_\d{2}_\d{3}$", subj):
            continue
        # Check if this patient is in the test set
        if not any(tid in subj for tid in test_patients):
            continue
        ses = os.path.join(deriv_root, subj, "ses-01")
        ctp_path = os.path.join(ses, f"{subj}_ses-01_space-ncct_ctp.nii.gz")
        lbl_path = os.path.join(ses, f"{subj}_ses-01_space-ncct_ctp_lesion_msk_3class.nii.gz")
        if os.path.isfile(ctp_path):
            test_files.append((
                subj, ctp_path,
                lbl_path if os.path.isfile(lbl_path) else None,
            ))

    logger.info(f"  Found {len(test_files)} test patients: "
                f"{[t[0] for t in test_files]}")

    if not test_files:
        logger.error("  No test files found!")
        return

    # ---- Class names + Dice helper ----
    class_names = ["background", "brain", "penumbra", "core"]

    def _dice(pred, gt, c):
        p = (pred == c).astype(np.float64)
        g = (gt == c).astype(np.float64)
        inter = (p * g).sum()
        union = p.sum() + g.sum()
        return float((2 * inter + 1e-8) / (union + 1e-8))

    # ---- Run ensemble inference per patient ----
    all_metrics = []
    ps = config.data.patch_size
    stride = getattr(args, 'stride', 4)  # use train stride or default 4 for test
    stride = min(stride, ps)  # ensure stride <= patch_size

    for pid, ctp_path, lbl_path in test_files:
        logger.info(f"\n  [{pid}]")
        ctp_sitk = sitk.ReadImage(ctp_path)
        ctp = sitk.GetArrayFromImage(ctp_sitk).astype(np.float32)  # (T, Z, Y, X)
        T, Z, Y, X = ctp.shape
        logger.info(f"    shape: T={T}, Z={Z}, Y={Y}, X={X}")

        # Normalise (same as training)
        clip_min, clip_max = config.data.clip_min, config.data.clip_max
        ctp_norm = (ctp - clip_min) / (clip_max - clip_min + 1e-8)
        ctp_norm = np.clip(ctp_norm, 0.0, 1.0)

        C = config.data.num_classes
        prob_sum = np.zeros((C, Z, Y, X), dtype=np.float64)
        count = np.zeros((Z, Y, X), dtype=np.float64)

        # Valid slices
        slice_sums = np.sum(np.abs(ctp_norm), axis=(0, 2, 3))
        valid_z = np.where(slice_sums > 0)[0]

        ys = list(range(0, max(Y - ps + 1, 1), stride))
        xs = list(range(0, max(X - ps + 1, 1), stride))

        for model in models:
            with torch.no_grad():
                for z in valid_z:
                    patches, positions = [], []
                    for y in ys:
                        for x in xs:
                            patch = ctp_norm[:, z, y:y+ps, x:x+ps]
                            # Skip empty patches — assign background directly
                            if np.abs(patch).sum() < 1e-6:
                                prob_sum[0, z, y:y+ps, x:x+ps] += 1.0  # background
                                count[z, y:y+ps, x:x+ps] += 1
                                continue
                            patches.append(patch)
                            positions.append((y, x))
                    if not patches:
                        continue
                    patches_np = np.stack(patches)
                    for i in range(0, len(patches_np), 2048):
                        batch = torch.from_numpy(patches_np[i:i+2048]).float().unsqueeze(1).to(device)
                        logits = model(batch)
                        probs = F.softmax(logits, dim=1).cpu().numpy()
                        for j, (y, x) in enumerate(positions[i:i+2048]):
                            prob_sum[:, z, y:y+ps, x:x+ps] += probs[j]
                            count[z, y:y+ps, x:x+ps] += 1

        # For uncovered voxels (count==0), set background probability to 1
        uncovered = (count < 1e-8)
        prob_sum[0][uncovered] = 1.0
        count[uncovered] = 1.0

        mean_probs = (prob_sum / count[np.newaxis]).astype(np.float32)
        seg = np.argmax(mean_probs, axis=0).astype(np.int16)

        # Post-process: remove small connected components (3-D)
        from Utils.plotting import postprocess_segmentation
        seg = postprocess_segmentation(seg, min_component_size=50)

        # ---- Save NIfTI prediction ----
        if lbl_path:
            ref_sitk = sitk.ReadImage(lbl_path)
        else:
            ref_size = list(ctp_sitk.GetSize())[:3]
            ref_spacing = list(ctp_sitk.GetSpacing())[:3]
            ref_origin = list(ctp_sitk.GetOrigin())[:3]
            ref_dir = ctp_sitk.GetDirection()
            dir_3x3 = list(ref_dir[:9]) if len(ref_dir) == 9 else list(np.array(ref_dir).reshape(4,4)[:3,:3].flatten())
            ref_sitk = sitk.Image(ref_size, sitk.sitkInt16)
            ref_sitk.SetSpacing(ref_spacing); ref_sitk.SetOrigin(ref_origin); ref_sitk.SetDirection(dir_3x3)

        seg_sitk = sitk.GetImageFromArray(seg)
        seg_sitk.CopyInformation(ref_sitk)
        nifti_path = os.path.join(test_dir, f"{pid}_prediction.nii.gz")
        sitk.WriteImage(seg_sitk, nifti_path)
        logger.info(f"    NIfTI saved: {nifti_path}")

        # Class distribution
        unique, counts_arr = np.unique(seg, return_counts=True)
        for u, c in zip(unique, counts_arr):
            logger.info(f"      Class {u}: {c:,} ({100*c/seg.size:.1f}%)")

        # ---- Metrics ----
        metrics = {}
        if lbl_path:
            gt = sitk.GetArrayFromImage(sitk.ReadImage(lbl_path)).astype(np.int16)
            for c_idx in range(C):
                metrics[f"dice_{class_names[c_idx]}"] = _dice(seg, gt, c_idx)
            metrics["dice_foreground"] = np.mean([metrics[f"dice_{class_names[c_idx]}"]
                                                   for c_idx in range(1, C)])
            p_les = ((seg == 2) | (seg == 3)).astype(np.float64)
            g_les = ((gt == 2) | (gt == 3)).astype(np.float64)
            inter = (p_les * g_les).sum(); union = p_les.sum() + g_les.sum()
            metrics["dice_lesion"] = float((2 * inter + 1e-8) / (union + 1e-8))
            all_metrics.append((pid, metrics))

            logger.info(f"    Dice — Brain: {metrics['dice_brain']:.4f}, "
                        f"Pen: {metrics['dice_penumbra']:.4f}, "
                        f"Core: {metrics['dice_core']:.4f}, "
                        f"Lesion: {metrics['dice_lesion']:.4f}")

        # ---- Save PNG slices with GT contours ----
        import cv2
        png_dir = os.path.join(test_dir, f"CTP_{pid}")
        tmp_dir = os.path.join(png_dir, "TMP")
        gt_dir  = os.path.join(png_dir, "GT")
        for d in [png_dir, tmp_dir, gt_dir]:
            os.makedirs(d, exist_ok=True)

        # Pick slices with most lesion content (pred or gt)
        slice_scores = []
        for z in range(Z):
            score = float((seg[z] == 2).sum() + 10 * (seg[z] == 3).sum())
            if lbl_path:
                score += float((gt[z] == 2).sum() + 10 * (gt[z] == 3).sum())
            slice_scores.append(score)
        top_slices = np.argsort(slice_scores)[::-1][:10]  # top 10 slices

        from Utils.plotting import REF_PIXEL_VALUES
        for z in top_slices:
            idx_str = f"{z:02d}"
            # Grayscale prediction
            pred_img = np.zeros((Y, X), dtype=np.uint8)
            for c_idx, pv in enumerate(REF_PIXEL_VALUES):
                pred_img[seg[z] == c_idx] = pv
            cv2.imwrite(os.path.join(png_dir, f"{idx_str}.png"), pred_img)

            if lbl_path:
                # GT image
                gt_img = np.zeros((Y, X), dtype=np.uint8)
                for c_idx, pv in enumerate(REF_PIXEL_VALUES):
                    gt_img[gt[z] == c_idx] = pv
                cv2.imwrite(os.path.join(gt_dir, f"{idx_str}.tiff"), gt_img)

                # Prediction + GT contours overlay
                pred_rgb = cv2.cvtColor(pred_img, cv2.COLOR_GRAY2RGB)
                # Penumbra GT contours (blue)
                pen_mask = ((gt_img >= 60) & (gt_img < 135)).astype(np.uint8) * 255
                pen_cnt, _ = cv2.findContours(pen_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(pred_rgb, pen_cnt, -1, (255, 0, 0), 2)
                # Core GT contours (red)
                core_mask = ((gt_img >= 135) & (gt_img < 234)).astype(np.uint8) * 255
                core_cnt, _ = cv2.findContours(core_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(pred_rgb, core_cnt, -1, (0, 0, 255), 2)
                cv2.imwrite(os.path.join(tmp_dir, f"{idx_str}.png"), pred_rgb)

        logger.info(f"    PNGs saved: {png_dir}/ ({len(top_slices)} slices)")

    # ---- Aggregate test metrics ----
    if all_metrics:
        logger.info("\n" + "=" * 60)
        logger.info("  Test Set Aggregate Metrics")
        logger.info("=" * 60)
        metric_keys = list(all_metrics[0][1].keys())
        for key in metric_keys:
            vals = [m[key] for _, m in all_metrics]
            logger.info(f"    {key:<25s}  {np.mean(vals):.4f} ± {np.std(vals):.4f}")

        # Save CSV
        csv_path = os.path.join(test_dir, "test_metrics.csv")
        with open(csv_path, "w") as f:
            f.write("patient," + ",".join(metric_keys) + "\n")
            for pid, m in all_metrics:
                f.write(f"{pid}," + ",".join(f"{m[k]:.6f}" for k in metric_keys) + "\n")
            f.write("MEAN," + ",".join(f"{np.mean([m[k] for _, m in all_metrics]):.6f}"
                                        for k in metric_keys) + "\n")
            f.write("STD," + ",".join(f"{np.std([m[k] for _, m in all_metrics]):.6f}"
                                       for k in metric_keys) + "\n")
        logger.info(f"  Test metrics CSV: {csv_path}")

    logger.info(f"  Test predictions saved to: {test_dir}")


def train_cv(config: Config, args: argparse.Namespace):
    """K-fold cross-validation training.

    Each fold trains a fresh model.  Best checkpoint and training curves
    are saved per fold, and aggregate metrics are reported at the end.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! This script requires GPU.")

    num_folds = config.train.num_folds
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv_dir = os.path.join(config.train.checkpoint_dir,
                          f"cv_{num_folds}fold_{run_timestamp}")
    os.makedirs(cv_dir, exist_ok=True)

    cv_logger = setup_logging(os.path.join(cv_dir, "logs"))
    cv_logger.info("=" * 60)
    cv_logger.info(f"  {num_folds}-fold Cross-Validation")
    cv_logger.info(f"  CV directory: {cv_dir}")
    cv_logger.info("=" * 60)
    cv_logger.info(f"Using device: cuda:0")
    cv_logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

    test_patients = getattr(config.data, 'test_patients', None)
    if test_patients:
        cv_logger.info(f"Holding out test patients: {test_patients}")

    fold_results = []
    for fold_idx, train_loader, val_loader in get_cv_fold_loaders(
        config=config,
        num_folds=num_folds,
        derivatives_dir=getattr(args, 'derivatives_dir', None),
        data_dir=getattr(args, 'data_dir', None),
        label_dir=getattr(args, 'label_dir', None),
        test_patients=test_patients,
        num_patients=getattr(args, 'num_patients', None),
    ):
        fold_dir = os.path.join(cv_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)
        fold_logger = setup_logging(os.path.join(fold_dir, "logs"))

        fold_logger.info("=" * 60)
        fold_logger.info(f"  Fold {fold_idx + 1}/{num_folds}")
        fold_logger.info("=" * 60)

        result = _run_training(config, args, train_loader, val_loader,
                               fold_dir, fold_logger)
        fold_results.append(result)

        cv_logger.info(
            f"Fold {fold_idx + 1}/{num_folds} done — "
            f"best_val_loss={result['best_val_loss']:.4f}, "
            f"best_fg_dice={result['best_foreground_dice']:.4f}"
        )

    # ---- Aggregate across folds ----
    cv_logger.info("\n" + "=" * 60)
    cv_logger.info("  Cross-Validation Summary")
    cv_logger.info("=" * 60)

    for key in ["best_val_loss", "best_foreground_dice",
                "best_dice_penumbra", "best_dice_core"]:
        values = [r[key] for r in fold_results]
        mean = np.mean(values)
        std  = np.std(values)
        cv_logger.info(f"  {key:<25s}  {mean:.4f} ± {std:.4f}  "
                       f"(per fold: {[f'{v:.4f}' for v in values]})")

    cv_logger.info("=" * 60)
    cv_logger.info(f"Results saved in: {cv_dir}")

    # ---- Held-out test set evaluation (ensemble across folds) ----
    test_patients = getattr(config.data, 'test_patients', None)
    if test_patients:
        cv_logger.info("\n" + "=" * 60)
        cv_logger.info("  Held-out Test Set Evaluation (Ensemble)")
        cv_logger.info("=" * 60)
        try:
            _evaluate_test_set(
                config=config,
                args=args,
                cv_dir=cv_dir,
                test_patients=test_patients,
                num_folds=num_folds,
                logger=cv_logger,
            )
        except Exception as e:
            cv_logger.error(f"  Test set evaluation failed: {e}")
            import traceback
            cv_logger.error(traceback.format_exc())
    else:
        cv_logger.info("\n  No test patients specified — skipping held-out evaluation.")


def main():
    parser = argparse.ArgumentParser(description="Train mJNet on 4D CTP data")
    
    # Data paths
    parser.add_argument(
        "--derivatives_dir",
        type=str,
        default=os.environ.get("SUS_CURATED_ROOT", "/path/to/Curated-SUS-2025"),
        help="Path to curated dataset root (contains derivatives/ folder). "
             "CTP and 3-class labels are auto-discovered.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="(Legacy) Path to directory containing 4D CTP NIfTI files. "
             "Overrides --derivatives_dir when combined with --label_dir.",
    )
    parser.add_argument(
        "--label_dir",
        type=str,
        default=None,
        help="(Legacy) Path to directory containing label NIfTI files. "
             "Overrides --derivatives_dir when combined with --data_dir.",
    )
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--loss", type=str, default="dice", 
                        choices=["dice", "squared_dice", "dice_standard", "focal", "tversky", "focal_tversky", "ce", "combined"],
                        help="Loss function. 'dice' (default) uses SquaredDiceLoss matching reference TF code.")
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["adam", "adamw", "sgd"],
                        help="Optimizer. SGD with Nesterov momentum recommended for segmentation.")
    parser.add_argument("--scheduler", type=str, default="step",
                        choices=["step", "cosine", "plateau", "none"],
                        help="LR scheduler. 'step' (default) decays by 0.1 every 10 epochs (matching reference).")
    
    # Model variant
    parser.add_argument("--longJ", action="store_true", help="Use longJ variant")
    parser.add_argument("--v2", action="store_true", help="Use v2 variant")
    
    # Training options
    parser.add_argument("--amp", action="store_true", help="Use mixed precision training")
    parser.add_argument("--no-augmentation", action="store_true", 
                        help="Disable data augmentation (rotation, flipping)")
    parser.add_argument("--dropout", action="store_true",
                        help="Enable Dropout3d at designated locations in the model")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay for optimizer (default: 1e-4)")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (default: 15)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", 
                        help="Directory to save checkpoints")
    
    # Data options
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--num_patients", type=int, default=None,
                        help="Limit total number of patients (for quick test runs). Uses all if not set.")
    parser.add_argument("--patch_size", type=int, default=16,
                        help="Spatial patch size (default: 16)")
    parser.add_argument("--stride", type=int, default=4,
                        help="Stride for patch extraction (default: 4). Use patch_size for non-overlapping.")
    parser.add_argument("--cache_dir", type=str, default="Data/npz_cache",
                        help="Directory to cache preprocessed .npz files (speeds up loading after first run)")
    parser.add_argument("--test_patients", type=str, nargs="*", default=None,
                        help="Patient IDs to hold out for testing (e.g., --test_patients 005 010). These won't be used in training or validation.")
    parser.add_argument("--oversample_lesion", type=int, default=6,
                        help="Oversample patches containing lesion (penumbra/core). N means lesion patches appear N times per epoch. 1=no oversampling, 4-8 recommended.")
    parser.add_argument("--class_weights", type=float, nargs=4, default=None,
                        help="Per-class loss weights [bg, brain, pen, core]. "
                             "For focal_tversky these act as a binary mask (0=exclude, >0=include). "
                             "Default: 0 1 1 1 (background excluded).")
    parser.add_argument("--num_folds", type=int, default=1,
                        help="Number of cross-validation folds. 1=single train/val split (default), "
                             "5=5-fold CV, etc. Each fold trains a fresh model.")
    parser.add_argument("--steps_per_epoch", type=int, default=0,
                        help="Max training batches per epoch. 0=use full dataset (default). "
                             "Useful to cap epoch length when dataset is very large.")
    
    args = parser.parse_args()
    
    # Create config
    config = Config()
    config.train.epochs = args.epochs
    config.train.batch_size = args.batch_size
    config.train.learning_rate = args.lr
    config.train.loss_function = args.loss
    config.train.optimizer = args.optimizer
    config.train.scheduler = args.scheduler  # LR scheduler
    config.train.use_longJ = args.longJ
    config.train.use_v2 = args.v2
    config.train.use_amp = args.amp
    config.train.augmentation = not args.no_augmentation  # Disable if flag is set
    config.train.dropout = args.dropout
    config.train.weight_decay = args.weight_decay
    config.train.patience = args.patience
    config.train.early_stopping_patience = args.patience
    config.train.checkpoint_dir = args.checkpoint_dir
    config.data.val_split = args.val_split
    config.data.num_workers = args.num_workers
    config.data.patch_size = args.patch_size
    config.data.stride = args.stride
    config.data.cache_dir = args.cache_dir
    config.data.oversample_lesion = args.oversample_lesion
    
    # Class weights — CLI overrides config default
    if args.class_weights is not None:
        config.data.class_weights = args.class_weights
    
    config.train.num_folds = args.num_folds
    config.train.steps_per_epoch = args.steps_per_epoch
    
    # Handle test patients - use CLI arg if provided, otherwise use config default
    if args.test_patients is not None:
        config.data.test_patients = args.test_patients
    
    # Create checkpoint directory
    os.makedirs(config.train.checkpoint_dir, exist_ok=True)
    
    # Train (single split or k-fold CV)
    if config.train.num_folds > 1:
        train_cv(config, args)
    else:
        train(config, args)


if __name__ == "__main__":
    main()
