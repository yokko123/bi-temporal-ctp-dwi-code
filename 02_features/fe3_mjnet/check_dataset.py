#!/usr/bin/env python
"""Quick sanity check for the lazy-loading mJ-Net dataset.

Not a pytest test, despite what this file used to be called: it needs the real
cohort on disk and has no assertions. Run it directly.

    SUS_CURATED_ROOT=/path/to/Curated-SUS-2025 python check_dataset.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Data.dataset import get_data_loaders
from config import Config


def main():
    cfg = Config()
    cfg.data.cache_dir = "/tmp/mjnet_npz_test"
    cfg.data.patch_size = 16
    cfg.data.stride = 8

    derivatives_dir = os.environ.get("SUS_CURATED_ROOT", "/path/to/Curated-SUS-2025")

    t0 = time.time()
    train_loader, val_loader = get_data_loaders(
        cfg, derivatives_dir=derivatives_dir, num_patients=2
    )
    print(f"Data loaders created in {time.time() - t0:.1f}s")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    t1 = time.time()
    for ctp, lbl in train_loader:
        print(f"Batch shape: CTP={ctp.shape}, Label={lbl.shape}")
        print(f"CTP range: [{ctp.min():.3f}, {ctp.max():.3f}]")
        print(f"Label classes: {lbl.unique().tolist()}")
        print(f"First batch loaded in {time.time() - t1:.2f}s")
        break

    print("DONE")


if __name__ == "__main__":
    main()
