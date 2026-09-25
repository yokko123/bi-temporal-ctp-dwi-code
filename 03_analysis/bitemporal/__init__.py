"""Analysis code for *Bi-temporal Image-driven Acute Stroke Evolution Analysis*.

Stage 03 of the pipeline: turns the per-ROI feature tables written by stage 02
into the region-pair statistics, the ablation, the subgroup analysis and the
figures of the paper.

Entry points are the ``run_*.py`` scripts one directory up.
"""

from .rois import REGIONS, TESTS, TEST_PAIRS, FE1_FEATURES, FE2_FEATURES
from .effects import cliffs_delta, region_pair_tests
from .separation import patient_similarities, delta_cos, separation_tests, run_separation
from .data import load_features, to_patient_level, add_subgroups

__all__ = [
    "REGIONS", "TESTS", "TEST_PAIRS", "FE1_FEATURES", "FE2_FEATURES",
    "cliffs_delta", "region_pair_tests",
    "patient_similarities", "delta_cos", "separation_tests", "run_separation",
    "load_features", "to_patient_level", "add_subgroups",
]
