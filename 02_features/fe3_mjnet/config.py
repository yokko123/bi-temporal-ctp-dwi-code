"""
Configuration file for mJNet PyTorch training/testing.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import torch


@dataclass
class DataConfig:
    """Data configuration."""
    # Paths - UPDATE THESE TO YOUR DATA LOCATIONS
    ctp_dir: str = "/path/to/ctp/data"  # Directory containing 4D CTP NIfTI files
    label_dir: str = "/path/to/labels"   # Directory containing label NIfTI files
    output_dir: str = "./checkpoints"
    
    # Data parameters
    num_timepoints: int = 40  # Number of timepoints in CTP
    timepoints: int = 40      # Alias for num_timepoints
    patch_size: int = 16      # Patch size (M x N)
    stride: int = 4           # Stride for patch extraction
    
    # Image dimensions
    image_height: int = 512
    image_width: int = 512
    
    # Class configuration
    num_classes: int = 4
    class_names: List[str] = field(default_factory=lambda: ["background", "brain", "penumbra", "core"])
    class_values: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    
    # Class weights for imbalanced data [background, brain, penumbra, core]
    # For focal_tversky these are a BINARY MASK: 0=exclude, >0=include.
    # Background is excluded (0.0) so the loss focuses entirely on brain,
    # penumbra, and core.  Class imbalance is handled by alpha/beta.
    class_weights: List[float] = field(default_factory=lambda: [0.0, 1.0, 1.0, 1.0])
    
    # Oversample patches containing lesion (penumbra/core) to combat class imbalance.
    # Value of N means lesion patches appear N times in each epoch.
    # 1 = no oversampling, 4-8 recommended for severe imbalance.
    oversample_lesion: int = 6
    
    # Normalization
    normalize: bool = True
    clip_min: float = 0.0
    clip_max: float = 255.0  # gray-level range for CTP
    
    # Train/val/test split
    val_split: float = 0.2
    random_seed: int = 42
    num_workers: int = 4
    
    # Test patients to exclude from training/validation (held out for final evaluation)
    # Format: list of patient ID substrings, e.g. ["sub-01_005", "sub-01_010"]
    test_patients: List[str] = field(default_factory=list)


@dataclass
class TrainConfig:
    """Training configuration."""
    # Training parameters
    epochs: int = 80
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    
    # Optimizer
    optimizer: str = "adam"  # "adam", "adamw", or "sgd"
    momentum: float = 0.9    # For SGD
    beta1: float = 0.9       # For Adam
    beta2: float = 0.999     # For Adam
    
    # Learning rate scheduler
    scheduler: str = "plateau"  # "cosine", "plateau", "step", or "none"
    lr_scheduler: str = "reduce_on_plateau"  # Legacy alias
    lr_patience: int = 5
    lr_factor: float = 0.1
    lr_step_size: int = 10
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 5  # Early stopping patience
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-5
    
    # Steps per epoch (0 = use full dataset)
    steps_per_epoch: int = 0
    
    # Validation
    val_split: float = 0.1  # Fraction of data for validation
    num_val_patients: int = 0  # If > 0, use specific number of patients for validation
    num_folds: int = 1  # 1=single split, >1=k-fold cross-validation
    
    # Data augmentation
    augmentation: bool = True
    
    # Loss function
    loss_function: str = "dice"  # "dice", "ce", "focal", "tversky", "combined"
    
    # Model variant
    model_variant: str = "standard"  # "standard", "longJ", "v2"
    use_longJ: bool = False
    use_v2: bool = False
    batch_norm: bool = True
    dropout: bool = False
    dropout_rates: Dict[str, float] = field(default_factory=lambda: {
        'long.1': 0.2, 'long.2': 0.2, 'long.3': 0.2,
        '1': 0.2, '2': 0.2, '3': 0.2, '4': 0.2, '5': 0.2
    })
    
    # Hardware
    num_workers: int = 4
    pin_memory: bool = True
    use_amp: bool = False  # Use AMP for faster training
    mixed_precision: bool = False  # Legacy alias for use_amp
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5  # Save checkpoint every N epochs
    save_best_only: bool = True
    
    # Reproducibility
    seed: int = 42


@dataclass 
class TestConfig:
    """Testing/inference configuration."""
    # Model checkpoint
    checkpoint_path: str = ""
    
    # Testing parameters
    batch_size: int = 64
    
    # Output
    save_predictions: bool = True
    output_dir: str = "./predictions"
    
    # Evaluation
    compute_metrics: bool = True
    
    # Hardware
    num_workers: int = 4


class Config:
    """Main configuration class."""
    
    def __init__(self):
        self.data = DataConfig()
        self.train = TrainConfig()
        self.test = TestConfig()
        
    def update_from_args(self, args):
        """Update config from command line arguments."""
        for key, value in vars(args).items():
            if hasattr(self.data, key):
                setattr(self.data, key, value)
            elif hasattr(self.train, key):
                setattr(self.train, key, value)
            elif hasattr(self.test, key):
                setattr(self.test, key, value)
    
    def get_device(self):
        """Get the device to use for training/testing."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def print_config(self):
        """Print configuration."""
        print("=" * 60)
        print("Configuration")
        print("=" * 60)
        print("\nData Config:")
        for k, v in vars(self.data).items():
            print(f"  {k}: {v}")
        print("\nTrain Config:")
        for k, v in vars(self.train).items():
            print(f"  {k}: {v}")
        print("\nTest Config:")
        for k, v in vars(self.test).items():
            print(f"  {k}: {v}")
        print("=" * 60)


# Default configuration instance
default_config = Config()
