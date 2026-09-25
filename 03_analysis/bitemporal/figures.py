"""t-SNE projections (Figs. 2 and 4) and bubble plots (Fig. 3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from .rois import REGIONS

#: Fate colouring used throughout the paper: green = survived, red = infarcted.
FATE_COLORS = {
    "clb_brain": "#009E73",
    "pen_brain": "#009E73",
    "core_brain": "#009E73",
    "core_fi": "#FF0000",
    "pen_fi": "#FF0000",
    "nhb_fi": "#FF0000",
}
MARKERS = {
    "clb_brain": "o",
    "nhb_fi": "o",
    "pen_brain": "s",
    "pen_fi": "s",
    "core_brain": "X",
    "core_fi": "X",
}


def tsne_embedding(
    df: pd.DataFrame,
    feature_cols: list[str],
    random_state: int = 42,
) -> pd.DataFrame:
    """Two-dimensional t-SNE of z-scored features, with the paper's settings.

    Perplexity follows ``min(30, n/3)``, which keeps small strata projectable.
    """
    x = df[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.mean())
    x = StandardScaler().fit_transform(x.to_numpy())

    n = len(df)
    perplexity = max(2, min(30, n // 3)) if n > 10 else max(2, n - 1)
    coords = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    ).fit_transform(x)

    out = df[[c for c in ("patient_id", "slice", "region") if c in df.columns]].copy()
    out["tsne_1"], out["tsne_2"] = coords[:, 0], coords[:, 1]
    return out


def plot_tsne(embedding: pd.DataFrame, title: str, out_path: str | Path, regions=None):
    """Scatter one t-SNE embedding, coloured by tissue fate."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regions = regions or REGIONS
    fig, ax = plt.subplots(figsize=(7, 5))
    for region in regions:
        sub = embedding[embedding["region"] == region]
        if sub.empty:
            continue
        ax.scatter(
            sub["tsne_1"], sub["tsne_2"],
            c=FATE_COLORS.get(region, "#777777"), marker=MARKERS.get(region, "o"),
            s=22, alpha=0.85, linewidths=0.4, edgecolors="black", label=region,
        )
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_bubbles(
    df: pd.DataFrame,
    features: list[str],
    out_path: str | Path,
    group_col: str | None = None,
    size_col: str = "n_voxels",
    title: str = "",
):
    """Median feature value per ROI class, bubble area encoding voxel count.

    With ``group_col`` set (the paper uses LVO vs non-LVO), each group is drawn
    as its own series: filled and larger for the first group, open and smaller
    for the second.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [None] if group_col is None else list(pd.unique(df[group_col].dropna()))
    present = [r for r in REGIONS if r in set(df["region"])]

    fig, axes = plt.subplots(1, len(features), figsize=(3.2 * len(features), 3.6), squeeze=False)
    for ax, feature in zip(axes[0], features):
        for gi, group in enumerate(groups):
            sub = df if group is None else df[df[group_col] == group]
            medians = sub.groupby("region")[feature].median()
            sizes = (
                sub.groupby("region")[size_col].median()
                if size_col in sub.columns
                else pd.Series(1.0, index=medians.index)
            )
            x = np.arange(len(present))
            y = [medians.get(r, np.nan) for r in present]
            area = np.array([sizes.get(r, 1.0) for r in present], dtype=float)
            area = 40 + 360 * (area / np.nanmax(area)) if np.nanmax(area) > 0 else np.full(len(x), 60.0)
            ax.scatter(
                x, y, s=area * (1.0 if gi == 0 else 0.5),
                c=[FATE_COLORS.get(r, "#777777") for r in present] if gi == 0 else "none",
                edgecolors=[FATE_COLORS.get(r, "#777777") for r in present],
                alpha=0.85 if gi == 0 else 1.0, linewidths=1.2,
                label=str(group) if group is not None else None,
            )
        ax.set_xticks(np.arange(len(present)))
        ax.set_xticklabels(present, rotation=45, ha="right", fontsize=7)
        ax.set_title(feature, fontsize=10)
    if group_col is not None:
        axes[0][0].legend(loc="best", fontsize=7, frameon=False)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path
