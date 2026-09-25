"""The cosine separation index Delta_cos for the CNN embedding families.

For a patient ``pt`` and two ROI classes R1, R2, using the slice-level embeddings
F^CNN_{pt,ROI,z}:

    Delta_cos(pt) = 1/2 [ sim(R1, R1) + sim(R2, R2) ] - sim(R1, R2)

where ``sim`` is the mean *absolute* cosine similarity over all slice pairs
(within-class pairs are taken without repetition and without the diagonal).
Higher values mean the two classes are better separated in embedding space.

Tables report the across-patient median; significance is a one-sample Wilcoxon
signed-rank test of Delta_cos against zero.

The pairwise similarities are evaluated as one Gram matrix per patient and
class block, which is exact and far faster than looping over slice pairs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .rois import REGIONS, TEST_PAIRS


def _unit_rows(x: np.ndarray, zscore: bool) -> np.ndarray:
    """Optionally z-score each feature over the cohort, then L2-normalise rows."""
    x = np.asarray(x, dtype=float)
    col_mean = np.nanmean(x, axis=0)
    x = np.where(np.isnan(x), col_mean, x)
    if zscore:
        sd = x.std(axis=0)
        sd[sd == 0] = 1.0
        x = (x - x.mean(axis=0)) / sd
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return x / norm


def patient_similarities(
    df: pd.DataFrame,
    feature_cols: list[str],
    zscore: bool = True,
    patient_col: str = "patient_id",
    region_col: str = "region",
    regions: list[str] | None = None,
    pairs: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Per-patient within-class and between-class mean absolute cosine similarity.

    ``df`` holds one row per (patient, slice, region). Normalisation is fitted
    once over the whole cohort passed in, so strata stay comparable.

    Returns one row per patient with ``<region>_self`` and ``<r1>__<r2>`` columns.
    A class present on fewer than two slices has no within-class similarity and
    yields NaN, which drops that patient from the affected tests.
    """
    regions = regions or REGIONS
    pairs = pairs or TEST_PAIRS

    unit = _unit_rows(df[feature_cols].to_numpy(), zscore=zscore)
    region_values = df[region_col].to_numpy()

    rows = []
    for patient, idx in df.groupby(patient_col, sort=True).indices.items():
        blocks = {r: unit[idx[region_values[idx] == r]] for r in regions}
        rec: dict[str, object] = {patient_col: patient}

        for region, block in blocks.items():
            if len(block) < 2:
                rec[f"{region}_self"] = np.nan
                continue
            gram = np.abs(block @ block.T)
            iu = np.triu_indices(len(block), k=1)
            rec[f"{region}_self"] = float(gram[iu].mean())

        for r1, r2 in pairs:
            a, b = blocks[r1], blocks[r2]
            rec[f"{r1}__{r2}"] = float(np.abs(a @ b.T).mean()) if len(a) and len(b) else np.nan

        rows.append(rec)

    return pd.DataFrame(rows)


def delta_cos(similarities: pd.DataFrame, r1: str, r2: str) -> pd.Series:
    """Per-patient Delta_cos for one region pair, patients with gaps dropped."""
    cols = [f"{r1}_self", f"{r2}_self", f"{r1}__{r2}"]
    sub = similarities[cols].dropna()
    return (sub[cols[0]] + sub[cols[1]]) / 2.0 - sub[cols[2]]


def separation_tests(
    similarities: pd.DataFrame,
    pairs: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Median Delta_cos and one-sample Wilcoxon signed-rank p per region pair."""
    pairs = pairs or TEST_PAIRS
    rows = []
    for r1, r2 in pairs:
        values = delta_cos(similarities, r1, r2)
        p = float(wilcoxon(values).pvalue) if len(values) else np.nan
        rows.append(
            {
                "pair": f"{r1}_vs_{r2}",
                "region_1": r1,
                "region_2": r2,
                "n_patients": int(len(values)),
                "median_delta_cos": float(np.median(values)) if len(values) else np.nan,
                "p_value": p,
                "significant": bool(p < 0.05) if np.isfinite(p) else False,
            }
        )
    return pd.DataFrame(rows)


def run_separation(
    df: pd.DataFrame,
    feature_cols: list[str],
    zscore: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Convenience wrapper: similarities then tests, in one call."""
    return separation_tests(
        patient_similarities(df, feature_cols, zscore=zscore, **kwargs),
        pairs=kwargs.get("pairs"),
    )
