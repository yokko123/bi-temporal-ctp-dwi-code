#!/usr/bin/env python3
"""Table V - subgroup analysis of the mJ-Net separation index (SUH cohort).

    python run_table5_subgroups.py --config config.yaml

Repeats the four region-pair tests inside each stratum:

* lesion hemisphere, left vs right (bilateral and midline cases excluded);
* onset-to-CT time, split at the median of the patients actually analysed.

Normalisation is fitted once over the whole cohort before the split, so the
strata stay on a common scale and their Delta_cos values remain comparable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import base_parser, load_family_slices, resolve_out_dir  # noqa: E402
from bitemporal.config import Config                                  # noqa: E402
from bitemporal.data import add_subgroups                             # noqa: E402
from bitemporal.separation import patient_similarities, separation_tests  # noqa: E402


def main() -> None:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--family", default="fe3", help="embedding family to stratify")
    args = parser.parse_args()

    config = Config.load(args.config)
    out_dir = resolve_out_dir(config, args.out_dir)

    df, feature_cols = load_family_slices(config, args.family, args.cohort)
    df, info = add_subgroups(
        df,
        clinical_csv=config.path(config.get("subgroups", "clinical_csv")),
        lesion_json=config.path(config.get("subgroups", "lesion_json")),
    )
    if "onset_to_ct_median_h" in info:
        print(f"onset-to-CT median split: {info['onset_to_ct_median_h']:.2f} h")

    strata = []
    if "hemisphere" in df.columns:
        for side in ("Left", "Right"):
            mask = df["hemisphere"].astype(str).str.lower().str.startswith(side.lower())
            strata.append((f"{side} hemisphere", df[mask]))
    if "onset_to_ct_bin" in df.columns:
        for value in sorted(df["onset_to_ct_bin"].dropna().unique()):
            strata.append((f"Onset-to-CT {value}", df[df["onset_to_ct_bin"] == value]))
    if not strata:
        raise SystemExit("no strata available; set subgroups.clinical_csv / subgroups.lesion_json")

    rows = []
    for name, subset in strata:
        if subset.empty:
            print(f"[{name}] empty, skipped")
            continue
        sep = separation_tests(patient_similarities(subset, feature_cols, zscore=True))
        sep.insert(0, "subgroup", name)
        sep.insert(1, "n_patients_in_subgroup", subset["patient_id"].nunique())
        rows.append(sep)
        print(f"[{name}] {subset['patient_id'].nunique()} patients")

    table = pd.concat(rows, ignore_index=True)
    table.to_csv(out_dir / f"table5_subgroups_{args.family}_{args.cohort}.csv", index=False)
    print(
        "\n"
        + table.pivot_table(index="subgroup", columns="pair", values="median_delta_cos")
        .round(3)
        .to_string()
    )
    print(f"\nwrote {out_dir}/table5_subgroups_{args.family}_{args.cohort}.csv")


if __name__ == "__main__":
    main()
