"""Effect sizes and significance testing for the handcrafted feature families.

Used for FE1 (baseline statistics) and FE2 (GLCM radiomics): a two-sided
Mann-Whitney U test per (region pair, feature), Benjamini-Hochberg FDR
correction, and Cliff's delta as the effect size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

#: Romano et al. thresholds for |Cliff's delta|.
CLIFF_THRESHOLDS = (0.147, 0.33, 0.474)


def cliffs_delta(x, y) -> tuple[float, str]:
    """Cliff's delta and its magnitude class.

    Returns ``delta`` in [-1, 1], positive when ``x`` dominates ``y``, together
    with one of ``negligible`` / ``small`` / ``medium`` / ``large``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return np.nan, "undefined"

    # Rank-based O(n log n) form of  (#{x>y} - #{x<y}) / (nx*ny).
    order = np.argsort(y, kind="mergesort")
    ys = y[order]
    greater = np.searchsorted(ys, x, side="left").sum()
    less = (y.size - np.searchsorted(ys, x, side="right")).sum()
    delta = float((greater - less) / (x.size * y.size))

    a = abs(delta)
    lo, mid, hi = CLIFF_THRESHOLDS
    magnitude = (
        "negligible" if a < lo else "small" if a < mid else "medium" if a < hi else "large"
    )
    return delta, magnitude


def region_pair_tests(
    df: pd.DataFrame,
    features: list[str],
    pairs: list[tuple[str, str]],
    region_col: str = "region",
    min_n: int = 4,
    fdr_scope: str = "per_pair",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Mann-Whitney U + Cliff's delta for every (region pair, feature).

    ``df`` must already be at the analysis unit of the paper: one row per
    patient and region (slice features max-pooled to patient level first).

    ``fdr_scope`` selects the Benjamini-Hochberg family:

    ``per_pair``
        Correct across the features of one region pair (m = len(features)).
        Reproduces the published FE1 (baseline) rows of Table III.
    ``per_family``
        Correct across every test of the feature family
        (m = len(pairs) * len(features)).
        Reproduces the published FE2 (GLCM) rows of Table III.

    The two published rows were produced by separately written notebooks and do
    not share a scope. Pick one scope deliberately.
    """
    if fdr_scope not in ("per_pair", "per_family"):
        raise ValueError(f"fdr_scope must be 'per_pair' or 'per_family', got {fdr_scope!r}")

    rows = []
    for r1, r2 in pairs:
        for feat in features:
            x = df.loc[df[region_col] == r1, feat].dropna().to_numpy()
            y = df.loc[df[region_col] == r2, feat].dropna().to_numpy()
            rec = {
                "pair": f"{r1}_vs_{r2}",
                "region_1": r1,
                "region_2": r2,
                "feature": feat,
                "n_1": x.size,
                "n_2": y.size,
            }
            if x.size < min_n or y.size < min_n:
                rec.update(p_raw=np.nan, cliffs_delta=np.nan, magnitude="too_small")
            else:
                _, p = mannwhitneyu(x, y, alternative="two-sided")
                delta, magnitude = cliffs_delta(x, y)
                rec.update(p_raw=float(p), cliffs_delta=delta, magnitude=magnitude)
            rows.append(rec)

    out = pd.DataFrame(rows)
    out["p_fdr"] = np.nan
    groups = out.groupby("pair").groups if fdr_scope == "per_pair" else {"all": out.index}
    for idx in groups.values():
        block = out.loc[idx]
        valid = block["p_raw"].notna()
        if valid.any():
            out.loc[block.index[valid], "p_fdr"] = multipletests(
                block.loc[valid, "p_raw"], method="fdr_bh"
            )[1]
    out["significant"] = out["p_fdr"].notna() & (out["p_fdr"] < alpha)
    out.attrs["fdr_scope"] = fdr_scope
    return out
