"""Bi-temporal ROI classes and the four region-pair tests of the paper.

The six outcome-aware classes come from intersecting the admission (T1) tissue
label with the follow-up (T2) outcome label:

    ROI^{T2}_{T1} := R_{T1} INTERSECT R_{T2}

    core_fi     ROI^{fi}_{c}     core, remained infarcted
    core_brain  ROI^{b}_{c}      core, salvaged
    pen_fi      ROI^{fi}_{p}     penumbra, progressed to infarct
    pen_brain   ROI^{b}_{p}      penumbra, salvaged
    clb_brain   ROI^{b}_{CLB}    contralateral healthy brain, remained healthy
    nhb_fi      ROI^{fi}_{NHB}   non-hypoperfused brain, later infarcted

Integer codes match the 6-class label maps written by the preprocessing stage
(01_preprocessing/post_reg_processing/p04_generate_6_class_labels.py).
"""

REGIONS = ["core_fi", "core_brain", "pen_fi", "pen_brain", "clb_brain", "nhb_fi"]

LABEL_CODES = {
    "core_fi": 1,
    "core_brain": 2,
    "pen_fi": 3,
    "pen_brain": 4,
    "clb_brain": 5,
    "nhb_fi": 6,
}

#: The four region-pair comparisons, in paper order (Tests 1-4).
#: Each entry is (test_id, region_1, region_2, short description).
TESTS = [
    ("test1", "pen_brain", "pen_fi", "salvaged vs infarcted penumbra"),
    ("test2", "core_brain", "core_fi", "salvaged vs infarcted core"),
    ("test3", "nhb_fi", "pen_fi", "infarcted non-hypoperfused brain vs infarcted penumbra"),
    ("test4", "nhb_fi", "clb_brain", "infarcted non-hypoperfused brain vs healthy contralateral"),
]

TEST_PAIRS = [(a, b) for _, a, b, _ in TESTS]

#: Display names used in the paper's tables and figures.
PRETTY = {
    "core_fi": "ROI^fi_c",
    "core_brain": "ROI^b_c",
    "pen_fi": "ROI^fi_p",
    "pen_brain": "ROI^b_p",
    "clb_brain": "ROI^b_CLB",
    "nhb_fi": "ROI^fi_NHB",
}

#: Feature names retained per handcrafted family (Table III row order).
FE1_FEATURES = ["mean", "std", "skewness", "min", "max", "kurtosis"]
FE2_FEATURES = ["Imc1", "Imc2", "Correlation", "MCC"]
