"""Rendering of the paper's tables from the statistics frames."""

from __future__ import annotations

import pandas as pd

from .rois import TESTS

_TICK, _CROSS = "\\checkmark", "$\\times$"
_ABBREV = {"negligible": "negl.", "small": "small", "medium": "med.", "large": "large"}


def _cells(stats: pd.DataFrame, feature: str) -> list[str]:
    cells = []
    for _, r1, r2, _ in TESTS:
        row = stats[(stats["feature"] == feature) & (stats["pair"] == f"{r1}_vs_{r2}")]
        if row.empty:
            cells.append("--")
            continue
        row = row.iloc[0]
        mark = _TICK if row["significant"] else _CROSS
        cells.append(f"{mark} {_ABBREV.get(row['magnitude'], row['magnitude'])}")
    return cells


def handcrafted_block(stats: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Table III rows for one handcrafted family: significance + effect class."""
    return pd.DataFrame(
        [[feature, *_cells(stats, feature)] for feature in features],
        columns=["Feature", *(f"Test {i}" for i in range(1, len(TESTS) + 1))],
    )


def separation_block(sep: pd.DataFrame, label: str) -> pd.DataFrame:
    """Table III row for one CNN family: significance + median Delta_cos."""
    cells = []
    for _, r1, r2, _ in TESTS:
        row = sep[sep["pair"] == f"{r1}_vs_{r2}"]
        if row.empty:
            cells.append("--")
            continue
        row = row.iloc[0]
        mark = _TICK if row["significant"] else _CROSS
        cells.append(f"{mark} {row['median_delta_cos']:.3f}")
    return pd.DataFrame(
        [[label, *cells]],
        columns=["Feature", *(f"Test {i}" for i in range(1, len(TESTS) + 1))],
    )


def to_latex(table: pd.DataFrame, caption: str, label: str) -> str:
    """A minimal booktabs-free tabular, ready to paste into the manuscript."""
    columns = "l" + "c" * (table.shape[1] - 1)
    lines = [
        "\\begin{table}[t]",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\centering",
        f"\\begin{{tabular}}{{{columns}}}",
        "\\hline",
        " & ".join(table.columns) + " \\\\",
        "\\hline",
    ]
    lines += [" & ".join(str(v) for v in row) + " \\\\" for row in table.itertuples(index=False)]
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)
