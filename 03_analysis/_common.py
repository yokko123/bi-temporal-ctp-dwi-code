"""Shared argument parsing and feature loading for the run_* scripts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bitemporal.config import Config          # noqa: E402
from bitemporal.data import load_features, to_patient_level  # noqa: E402


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="config.yaml", help="YAML config (see config.example.yaml)")
    parser.add_argument("--cohort", default="sus", choices=["sus", "isles"], help="which cohort to analyse")
    parser.add_argument("--out-dir", default=None, help="override out_dir from the config")
    return parser


def resolve_out_dir(config: Config, override: str | None) -> Path:
    out = Path(override) if override else config.out_dir
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_family(config: Config, family: str, cohort: str, aggregate: str = "max"):
    """Load one feature family and return ``(patient_level_frame, feature_cols)``."""
    entry = config.family(family, cohort)
    df, feature_cols = load_features(
        entry["path"],
        feature_prefix=entry.get("prefix"),
        feature_suffix=entry.get("suffix"),
        features=entry.get("columns"),
    )
    return to_patient_level(df, feature_cols, how=aggregate), feature_cols


def load_family_slices(config: Config, family: str, cohort: str):
    """Load one feature family at slice level (required for Delta_cos)."""
    entry = config.family(family, cohort)
    df, feature_cols = load_features(
        entry["path"],
        feature_prefix=entry.get("prefix"),
        feature_suffix=entry.get("suffix"),
        features=entry.get("columns"),
    )
    if "slice" not in df.columns:
        raise SystemExit(
            f"features.{family}.{cohort} is patient level; Delta_cos needs slice-level embeddings"
        )
    return df, feature_cols
