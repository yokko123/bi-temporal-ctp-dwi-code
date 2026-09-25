"""Loading and normalising the feature tables written by stage 02.

The four feature families were extracted by separate scripts and do not share a
column convention, so everything is funnelled through :func:`load_features`,
which returns a frame with the canonical columns ``patient_id``, ``region`` and,
for slice-level tables, ``slice``.

Canonical region names are those in :mod:`bitemporal.rois`.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .rois import REGIONS

_PATIENT_ALIASES = ("patient_id", "patient", "pid", "subject", "subject_id")
_REGION_ALIASES = ("region", "class_name", "roi", "label_name")
_SLICE_ALIASES = ("slice", "z", "z_index", "slice_index")

#: Columns that are never features, whatever a table calls them.
_METADATA_COLUMNS = {
    "patient_id", "patient", "slice", "z", "region", "class_name", "label",
    "split", "n_voxels", "cohort", "sex", "recanalization", "mrs_3months",
    "recan_label", "mrs_group", "hemisphere", "onset_to_ct_hours",
    "onset_to_ct_bin", "ht", "smoking", "anticoagulation",
}


def _pick(columns, aliases: tuple[str, ...]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    return None


def canonical_patient_id(value: str) -> str:
    """Normalise the several patient-id spellings to ``sub-stroke_XX_NNN``.

    The mJ-Net tables store ``01_001`` while the clinical table and the nnU-Net
    tables store ``sub-stroke_01_001``. ISLES ids (``sub-stroke0001``) are left
    untouched.
    """
    value = str(value).strip()
    if re.fullmatch(r"\d{2}_\d{3}", value):
        return f"sub-stroke_{value}"
    return value


def load_features(
    path: str | Path,
    feature_prefix: str | None = None,
    feature_suffix: str | None = None,
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Read a feature table and return ``(frame, feature_columns)``.

    One of ``feature_prefix`` (e.g. ``"feat_"`` for the CNN families),
    ``feature_suffix`` (e.g. ``"_max"`` for the nnU-Net pooled columns) or an
    explicit ``features`` list selects the feature columns. With none of them,
    every numeric non-metadata column is taken, which is what the handcrafted
    tables need.
    """
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

    patient_col = _pick(df.columns, _PATIENT_ALIASES)
    region_col = _pick(df.columns, _REGION_ALIASES)
    if patient_col is None or region_col is None:
        raise ValueError(f"{path.name}: could not find patient/region columns in {list(df.columns)[:10]}")

    rename = {patient_col: "patient_id", region_col: "region"}
    slice_col = _pick(df.columns, _SLICE_ALIASES)
    if slice_col is not None:
        rename[slice_col] = "slice"
    df = df.rename(columns=rename)

    df["patient_id"] = df["patient_id"].map(canonical_patient_id)
    df["region"] = df["region"].astype(str).str.strip().str.lower()
    df = df[df["region"].isin(REGIONS)].reset_index(drop=True)

    if features is not None:
        feature_cols = list(features)
    elif feature_prefix is not None:
        feature_cols = [c for c in df.columns if c.startswith(feature_prefix)]
    elif feature_suffix is not None:
        feature_cols = [c for c in df.columns if c.endswith(feature_suffix)]
    else:
        # No selector: take every numeric column that is not bookkeeping. This is
        # what the handcrafted tables need, where features are named outright.
        feature_cols = [
            c for c in df.columns
            if c not in _METADATA_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
        ]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing feature columns {missing[:5]}")
    if not feature_cols:
        raise ValueError(f"{path.name}: no feature columns matched")

    return df, feature_cols


def to_patient_level(
    df: pd.DataFrame,
    feature_cols: list[str],
    how: str = "max",
) -> pd.DataFrame:
    """Aggregate slice-level descriptors to one vector per patient and region.

    The paper aggregates by max-pooling across slices; ``how`` is exposed so the
    ablation can vary it. A table that is already patient-level passes through.
    """
    if "slice" not in df.columns:
        return df.copy()
    grouped = df.groupby(["patient_id", "region"], as_index=False)[feature_cols]
    return getattr(grouped, how)()


def add_subgroups(
    df: pd.DataFrame,
    clinical_csv: str | Path | None = None,
    lesion_json: str | Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Attach the Table V strata: lesion hemisphere and onset-to-CT timing.

    ``lesion_json`` maps ``"01_001" -> "Left"/"Right"`` (blank for bilateral or
    midline cases, which the paper excludes from the hemisphere analysis).
    ``clinical_csv`` must carry ``patient_id`` and ``onset_to_ct_time`` as
    ``H:MM[:SS]``. The timing split is the median over the patients present in
    ``df``, so it is defined on the cohort actually analysed.

    Returns the annotated frame and a dict of the split values used.
    """
    import json

    df = df.copy()
    info: dict = {}

    if lesion_json is not None:
        with open(lesion_json) as fh:
            lesion = json.load(fh)
        hemisphere = {canonical_patient_id(k): str(v).strip() for k, v in lesion.items()}
        df["hemisphere"] = df["patient_id"].map(hemisphere).replace("", np.nan)
        info["hemisphere_n"] = df.groupby("hemisphere")["patient_id"].nunique().to_dict()

    if clinical_csv is not None:
        clinical = pd.read_csv(clinical_csv)
        clinical["patient_id"] = clinical["patient_id"].map(canonical_patient_id)
        clinical["onset_to_ct_hours"] = clinical["onset_to_ct_time"].map(_parse_hms)
        present = set(df["patient_id"])
        median = float(clinical.loc[clinical["patient_id"].isin(present), "onset_to_ct_hours"].median())
        info["onset_to_ct_median_h"] = median
        hours = dict(zip(clinical["patient_id"], clinical["onset_to_ct_hours"]))
        df["onset_to_ct_hours"] = df["patient_id"].map(hours)
        # Patients without a recorded onset time stay NaN rather than being
        # swept into one side of the split; the paper excludes them.
        binned = pd.Series(pd.NA, index=df.index, dtype="object")
        known = df["onset_to_ct_hours"].notna()
        binned[known & (df["onset_to_ct_hours"] < median)] = f"< {median:.2f}h"
        binned[known & (df["onset_to_ct_hours"] >= median)] = f">= {median:.2f}h"
        df["onset_to_ct_bin"] = binned
        info["onset_to_ct_missing_patients"] = int(
            df.loc[~known, "patient_id"].nunique()
        )

    return df, info


def _parse_hms(value) -> float:
    """``"4:32:00"`` -> 4.533 hours; anything unparseable -> NaN."""
    try:
        parts = str(value).split(":")
        hours = int(parts[0])
        minutes = int(parts[1]) if len(parts) > 1 else 0
        seconds = int(parts[2]) if len(parts) > 2 else 0
        return hours + minutes / 60 + seconds / 3600
    except (ValueError, IndexError):
        return np.nan
