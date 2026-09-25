#!/usr/bin/env python3
"""Table IV - normalisation x pooling ablation for the CNN embeddings.

    python run_table4_ablation.py --config config.yaml

Crosses two knobs for FE3 (mJ-Net) and FE4 (nnU-Net):

* normalisation - raw embeddings vs cohort-wise z-scoring before the cosine;
* aggregation   - how encoder activations were pooled over the voxels of an ROI
  into one slice descriptor (mean / median / max).

The aggregation knob reads a different column family or file per operator, so
the config must point at all three. For nnU-Net one file carries ``f*_mean``,
``f*_median`` and ``f*_max``, so only the suffix changes; ``aggregation_paths``
overrides the file per operator when the family stores them separately.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import base_parser, resolve_out_dir          # noqa: E402
from bitemporal.config import Config                      # noqa: E402
from bitemporal.data import load_features                 # noqa: E402
from bitemporal.separation import patient_similarities, separation_tests  # noqa: E402

AGGREGATIONS = ("mean", "median", "max")


def _available_aggregations(entry: dict) -> list[str]:
    """Which pooling operators this family can actually be varied over.

    Either the family stores one file per operator (``aggregation_paths``), or
    one file carries a column family per operator (``suffix: _max`` and friends).
    With neither, the file holds a single pooling and the ablation degenerates to
    the one configured variant; reporting three identical rows would be a lie.
    """
    if entry.get("aggregation_paths"):
        return [a for a in AGGREGATIONS if a in entry["aggregation_paths"]]
    suffix = (entry.get("suffix") or "").lstrip("_")
    if suffix in AGGREGATIONS:
        return list(AGGREGATIONS)
    return ["as-configured"]


def _load_variant(config: Config, family: str, cohort: str, aggregation: str):
    entry = config.family(family, cohort)
    overrides = entry.get("aggregation_paths") or {}
    path = overrides.get(aggregation, entry["path"])

    suffix = entry.get("suffix")
    if suffix and suffix.lstrip("_") in AGGREGATIONS and aggregation in AGGREGATIONS:
        suffix = f"_{aggregation}"          # nnU-Net: same file, other column family
    return load_features(
        path,
        feature_prefix=entry.get("prefix"),
        feature_suffix=suffix,
        features=entry.get("columns"),
    )


def main() -> None:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--families", default="fe3,fe4", help="comma-separated subset")
    args = parser.parse_args()

    config = Config.load(args.config)
    out_dir = resolve_out_dir(config, args.out_dir)
    rows = []

    for family in [f.strip() for f in args.families.split(",") if f.strip()]:
        aggregations = _available_aggregations(config.family(family, args.cohort))
        if aggregations == ["as-configured"]:
            print(
                f"[{family}] only one pooling available; set features.{family}.{args.cohort}."
                "aggregation_paths to vary it"
            )
        for aggregation in aggregations:
            try:
                df, feature_cols = _load_variant(config, family, args.cohort, aggregation)
            except (SystemExit, FileNotFoundError, ValueError) as exc:
                print(f"[{family}/{aggregation}] skipped: {exc}")
                continue
            for zscore in (False, True):
                sep = separation_tests(patient_similarities(df, feature_cols, zscore=zscore))
                for _, row in sep.iterrows():
                    rows.append(
                        {
                            "family": family,
                            "normalisation": "z-score" if zscore else "none",
                            "aggregation": aggregation,
                            "pair": row["pair"],
                            "n_patients": row["n_patients"],
                            "median_delta_cos": row["median_delta_cos"],
                            "p_value": row["p_value"],
                            "significant": row["significant"],
                        }
                    )
            print(f"[{family}/{aggregation}] done ({len(feature_cols)} dims)")

    if not rows:
        raise SystemExit("no variant could be loaded; check aggregation_paths in the config")

    long = pd.DataFrame(rows)
    long.to_csv(out_dir / f"table4_ablation_{args.cohort}.csv", index=False)
    wide = long.pivot_table(
        index=["family", "normalisation", "aggregation"],
        columns="pair",
        values="median_delta_cos",
    ).round(3)
    print("\n" + wide.to_string())
    wide.to_csv(out_dir / f"table4_ablation_{args.cohort}_wide.csv")
    print(f"\nwrote {out_dir}/table4_ablation_{args.cohort}*.csv")


if __name__ == "__main__":
    main()
