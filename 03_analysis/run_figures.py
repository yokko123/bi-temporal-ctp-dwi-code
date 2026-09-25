#!/usr/bin/env python3
"""Figures 2-4 - t-SNE projections and the handcrafted bubble plots.

    python run_figures.py --config config.yaml --what tsne,bubble

* ``tsne``   Fig. 2 (all six ROI classes, one panel per feature family) and
             Fig. 4 (one panel per region-pair test, FE3 only).
* ``bubble`` Fig. 3 (median FE1 and FE2 values per ROI class, bubble area =
             voxel count, split by LVO / non-LVO when the column is present).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import base_parser, load_family, load_family_slices, resolve_out_dir  # noqa: E402
from bitemporal.config import Config                              # noqa: E402
from bitemporal.figures import plot_bubbles, plot_tsne, tsne_embedding  # noqa: E402
from bitemporal.rois import FE1_FEATURES, FE2_FEATURES, TESTS     # noqa: E402


def main() -> None:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--what", default="tsne,bubble", help="tsne, bubble or both")
    parser.add_argument("--families", default="fe1,fe2,fe3,fe4")
    args = parser.parse_args()

    config = Config.load(args.config)
    out_dir = resolve_out_dir(config, args.out_dir) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = {w.strip() for w in args.what.split(",")}
    families = [f.strip() for f in args.families.split(",") if f.strip()]

    if "tsne" in wanted:
        for family in families:
            try:
                df, feature_cols = load_family_slices(config, family, args.cohort)
            except SystemExit:
                df, feature_cols = load_family(config, family, args.cohort)
            embedding = tsne_embedding(df, feature_cols)
            plot_tsne(
                embedding,
                f"t-SNE, {family.upper()} ({args.cohort})",
                out_dir / f"fig2_tsne_{family}_{args.cohort}.png",
            )
            print(f"[tsne/{family}] wrote fig2_tsne_{family}_{args.cohort}.png")

            if family == "fe3":  # Fig. 4: one panel per region-pair test
                for test_id, r1, r2, _ in TESTS:
                    pair_df = df[df["region"].isin([r1, r2])]
                    plot_tsne(
                        tsne_embedding(pair_df, feature_cols),
                        f"{test_id}: {r1} vs {r2}",
                        out_dir / f"fig4_tsne_{test_id}_{args.cohort}.png",
                        regions=[r1, r2],
                    )
                print(f"[tsne/{family}] wrote fig4_tsne_test1..4_{args.cohort}.png")

    if "bubble" in wanted:
        for family, features in (("fe1", FE1_FEATURES), ("fe2", FE2_FEATURES)):
            if family not in families:
                continue
            df, _ = load_family(config, family, args.cohort)
            group_col = "cohort" if "cohort" in df.columns else None
            plot_bubbles(
                df, features,
                out_dir / f"fig3_bubble_{family}_{args.cohort}.png",
                group_col=group_col,
                title=f"{family.upper()} medians per ROI class ({args.cohort})",
            )
            print(f"[bubble/{family}] wrote fig3_bubble_{family}_{args.cohort}.png")


if __name__ == "__main__":
    main()
