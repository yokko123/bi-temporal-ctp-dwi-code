"""Unit tests for the statistics that back Tables III-V.

    python -m pytest tests -q
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "03_analysis"))

from bitemporal.effects import cliffs_delta, region_pair_tests
from bitemporal.rois import TEST_PAIRS
from bitemporal.separation import patient_similarities, separation_tests


def _brute_force_cliffs(x, y):
    gt = sum(np.sum(xi > np.asarray(y)) for xi in x)
    lt = sum(np.sum(xi < np.asarray(y)) for xi in x)
    return (gt - lt) / (len(x) * len(y))


def test_cliffs_delta_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.normal(size=rng.integers(5, 40))
        y = rng.normal(loc=0.4, size=rng.integers(5, 40))
        assert cliffs_delta(x, y)[0] == pytest.approx(_brute_force_cliffs(x, y))


def test_cliffs_delta_handles_ties_and_extremes():
    assert cliffs_delta([1, 1, 1], [1, 1, 1])[0] == 0.0
    assert cliffs_delta([5, 6, 7], [1, 2, 3]) == (1.0, "large")
    assert cliffs_delta([1, 2, 3], [5, 6, 7]) == (-1.0, "large")
    assert cliffs_delta([], [1, 2])[1] == "undefined"


def test_cliffs_magnitude_thresholds():
    # Romano bins: <0.147 negligible, <0.33 small, <0.474 medium, else large.
    assert cliffs_delta([1, 2], [1, 2])[1] == "negligible"
    assert cliffs_delta(np.arange(100), np.arange(100) + 25)[1] in {"small", "medium", "large"}


def _similarity_reference(df, feature_cols, zscore=True):
    """Slow, literal transcription of the definition, for cross-checking."""
    x = df[feature_cols].to_numpy(float)
    if zscore:
        sd = x.std(axis=0)
        sd[sd == 0] = 1.0
        x = (x - x.mean(axis=0)) / sd
    frame = df.copy()
    frame[feature_cols] = x

    def cos(a, b):
        return abs(float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    rows = []
    for patient, sub in frame.groupby("patient_id"):
        rec = {"patient_id": patient}
        blocks = {r: sub[sub["region"] == r][feature_cols].to_numpy() for r in sub["region"].unique()}
        for region, block in blocks.items():
            if len(block) < 2:
                continue
            rec[f"{region}_self"] = float(np.mean([cos(a, b) for a, b in combinations(block, 2)]))
        for r1, r2 in TEST_PAIRS:
            a, b = blocks.get(r1), blocks.get(r2)
            if a is None or b is None or not len(a) or not len(b):
                continue
            rec[f"{r1}__{r2}"] = float(np.mean([cos(u, v) for u in a for v in b]))
        rows.append(rec)
    return pd.DataFrame(rows)


def _toy_embeddings(seed=0, patients=6, dims=8):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(patients):
        for region in ["pen_brain", "pen_fi", "core_brain", "core_fi", "nhb_fi", "clb_brain"]:
            for z in range(4):
                rows.append(
                    {"patient_id": f"p{p}", "slice": z, "region": region,
                     **{f"feat_{i}": v for i, v in enumerate(rng.normal(size=dims))}}
                )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("zscore", [True, False])
def test_vectorised_similarities_match_pairwise_loop(zscore):
    df = _toy_embeddings()
    cols = [c for c in df.columns if c.startswith("feat_")]
    fast = patient_similarities(df, cols, zscore=zscore).set_index("patient_id").sort_index()
    slow = _similarity_reference(df, cols, zscore=zscore).set_index("patient_id").sort_index()
    shared = [c for c in fast.columns if c in slow.columns]
    assert shared
    np.testing.assert_allclose(fast[shared].to_numpy(float), slow[shared].to_numpy(float), rtol=1e-10, atol=1e-12)


def test_separation_index_is_positive_when_classes_are_planted_apart():
    rng = np.random.default_rng(1)
    dims = 16
    direction = rng.normal(size=dims)
    direction /= np.linalg.norm(direction)
    rows = []
    for p in range(15):
        for region, shift in [("pen_brain", 0.0), ("pen_fi", 4.0), ("core_brain", 0.0),
                              ("core_fi", 0.05), ("nhb_fi", 6.0), ("clb_brain", 0.0)]:
            for z in range(5):
                rows.append(
                    {"patient_id": f"p{p}", "slice": z, "region": region,
                     **{f"feat_{i}": v for i, v in enumerate(rng.normal(scale=0.3, size=dims) + shift * direction)}}
                )
    df = pd.DataFrame(rows)
    cols = [c for c in df.columns if c.startswith("feat_")]
    sep = separation_tests(patient_similarities(df, cols, zscore=True)).set_index("pair")

    # Well-separated classes must beat the near-identical core pair.
    assert sep.loc["pen_brain_vs_pen_fi", "median_delta_cos"] > sep.loc["core_brain_vs_core_fi", "median_delta_cos"]
    assert sep.loc["nhb_fi_vs_clb_brain", "median_delta_cos"] > 0
    assert sep.loc["pen_brain_vs_pen_fi", "significant"]


def test_patient_with_single_slice_in_a_class_is_dropped_not_crashed():
    df = _toy_embeddings(patients=5)
    cols = [c for c in df.columns if c.startswith("feat_")]
    df = df.drop(df[(df.patient_id == "p0") & (df.region == "pen_fi") & (df.slice > 0)].index)
    sim = patient_similarities(df, cols)
    assert np.isnan(sim.loc[sim.patient_id == "p0", "pen_fi_self"].iloc[0])
    sep = separation_tests(sim).set_index("pair")
    assert sep.loc["pen_brain_vs_pen_fi", "n_patients"] == 4
    assert sep.loc["core_brain_vs_core_fi", "n_patients"] == 5


def test_fdr_scope_changes_the_family_size_not_the_raw_p():
    rng = np.random.default_rng(2)
    rows = []
    for p in range(40):
        for region, shift in [("pen_brain", 0.0), ("pen_fi", 0.55), ("core_brain", 0.0),
                              ("core_fi", 0.05), ("nhb_fi", 2.0), ("clb_brain", 0.0)]:
            rows.append({"patient_id": f"p{p}", "region": region,
                         **{f: float(shift + rng.normal()) for f in ["a", "b", "c", "d", "e", "f"]}})
    df = pd.DataFrame(rows)
    features = ["a", "b", "c", "d", "e", "f"]
    per_pair = region_pair_tests(df, features, TEST_PAIRS, fdr_scope="per_pair")
    per_family = region_pair_tests(df, features, TEST_PAIRS, fdr_scope="per_family")

    np.testing.assert_allclose(per_pair["p_raw"], per_family["p_raw"])
    # The larger family can only make a given p-value's adjustment less liberal
    # in the small-m block, never change the underlying test.
    assert (per_pair["p_fdr"] >= per_pair["p_raw"] - 1e-12).all()
    assert (per_family["p_fdr"] >= per_family["p_raw"] - 1e-12).all()
    with pytest.raises(ValueError):
        region_pair_tests(df, features, TEST_PAIRS, fdr_scope="nonsense")


def test_region_pair_tests_marks_undersized_groups():
    df = pd.DataFrame(
        [{"patient_id": f"p{i}", "region": "pen_brain", "a": float(i)} for i in range(10)]
        + [{"patient_id": f"q{i}", "region": "pen_fi", "a": float(i)} for i in range(2)]
    )
    out = region_pair_tests(df, ["a"], [("pen_brain", "pen_fi")], min_n=4)
    assert out.iloc[0]["magnitude"] == "too_small"
    assert not out.iloc[0]["significant"]
