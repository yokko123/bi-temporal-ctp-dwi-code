#!/usr/bin/env python3
"""Table III - the four region-pair comparisons across FE1-FE4.

    python run_table3_region_pairs.py --config config.yaml --cohort sus

Writes one CSV per feature family plus a combined LaTeX table into ``out_dir``.
Everything is patient level: slice descriptors are max-pooled to
(patient x region) before any test is run.
"""

from __future__ import annotations

from _common import base_parser, load_family, load_family_slices, resolve_out_dir
from bitemporal.config import Config
from bitemporal.effects import region_pair_tests
from bitemporal.rois import FE1_FEATURES, FE2_FEATURES, TEST_PAIRS
from bitemporal.separation import patient_similarities, separation_tests
from bitemporal.tables import handcrafted_block, separation_block, to_latex

import pandas as pd


def main() -> None:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--families", default="fe1,fe2,fe3,fe4", help="comma-separated subset to run")
    args = parser.parse_args()

    config = Config.load(args.config)
    out_dir = resolve_out_dir(config, args.out_dir)
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    blocks = []

    for family, features in (("fe1", FE1_FEATURES), ("fe2", FE2_FEATURES)):
        if family not in families:
            continue
        df, _ = load_family(config, family, args.cohort)
        scope = config.get("fdr_scope", family, default="per_pair")
        stats = region_pair_tests(df, features, TEST_PAIRS, fdr_scope=scope)
        stats.to_csv(out_dir / f"table3_{family}_{args.cohort}.csv", index=False)
        print(f"[{family}] {df['patient_id'].nunique()} patients, FDR scope = {scope}")
        blocks.append(handcrafted_block(stats, features))

    for family, label in (("fe3", "Delta_cos (FE3 mJ-Net)"), ("fe4", "Delta_cos (FE4 nnU-Net)")):
        if family not in families:
            continue
        df, feature_cols = load_family_slices(config, family, args.cohort)
        similarities = patient_similarities(df, feature_cols, zscore=True)
        sep = separation_tests(similarities)
        sep.to_csv(out_dir / f"table3_{family}_{args.cohort}.csv", index=False)
        print(f"[{family}] {df['patient_id'].nunique()} patients, {len(feature_cols)} dims")
        blocks.append(separation_block(sep, label))

    if blocks:
        table = pd.concat(blocks, ignore_index=True)
        (out_dir / f"table3_{args.cohort}.tex").write_text(
            to_latex(table, "Region-pair comparisons.", f"tab:region_pairs_{args.cohort}")
        )
        print(table.to_string(index=False))
        print(f"\nwrote {out_dir}/table3_{args.cohort}.*")


if __name__ == "__main__":
    main()
