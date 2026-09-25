# Stage 03 - statistics and figures

Turns the feature tables from stage 02 into the paper's tables and figures.
This is the stage that was rewritten for release.

```bash
cp config.example.yaml config.yaml     # then fill in your paths
python run_table3_region_pairs.py --config config.yaml --cohort sus
python run_table4_ablation.py     --config config.yaml
python run_table5_subgroups.py    --config config.yaml
python run_figures.py             --config config.yaml
```

| Script | Produces |
|---|---|
| `run_table3_region_pairs.py` | Table III - the four region-pair comparisons across FE1-FE4 |
| `run_table4_ablation.py` | Table IV - normalisation x pooling ablation for the CNN embeddings |
| `run_table5_subgroups.py` | Table V - hemisphere and onset-to-CT strata |
| `run_figures.py` | Fig. 2 and 4 (t-SNE), Fig. 3 (bubble plots) |

## The package

`bitemporal/` holds one definition of each statistic, so all four feature
families are treated identically.

| Module | Contents |
|---|---|
| `rois.py` | the six classes, their label codes, the four tests |
| `data.py` | loading, column-name normalisation, patient-level pooling, subgroup strata |
| `effects.py` | Cliff's delta, Mann-Whitney U, Benjamini-Hochberg |
| `separation.py` | the cosine separation index and its Wilcoxon test |
| `tables.py` | LaTeX rendering |
| `figures.py` | t-SNE and bubble plots |
| `config.py` | the YAML config |

## The two statistics

**Handcrafted families (FE1, FE2).** Two-sided Mann-Whitney U per (region pair,
feature), Benjamini-Hochberg FDR at alpha = 0.05, Cliff's delta as effect size
with Romano's thresholds at 0.147 / 0.33 / 0.474.

The FDR family is a deliberate setting, not a constant, because the published
table does not use one consistently:

```yaml
fdr_scope:
  fe1: per_pair     # across the features of one region pair (m = 6)
  fe2: per_family   # across all pairs x features           (m = 16)
```

Those are the values that reproduce the paper. For new work, pick one for both.

**CNN families (FE3, FE4).** For each patient, the separation index

```
Delta_cos(pt) = 1/2 [ sim(R1,R1) + sim(R2,R2) ] - sim(R1,R2)
```

where `sim` is the mean absolute cosine similarity over slice pairs, computed on
embeddings z-scored once per cohort. Tables report the across-patient median;
significance is a one-sample Wilcoxon signed-rank test against zero. A patient
with fewer than two slices in a class contributes no within-class similarity and
is dropped from the affected test, which is why the per-test *n* is below the
cohort size.

Similarities are evaluated as one Gram matrix per patient and class block. This
is exact, not an approximation: `../tests/test_statistics.py` asserts agreement
with the literal pairwise loop to 1e-10.

## Aggregation

Everything is patient level. Slice descriptors are max-pooled to
(patient x region) before any test. `run_table4_ablation.py` is the one place
that varies the pooling, and it varies the *voxel-to-slice* pooling inside the
ROI, which is a property of the stage-02 tables: either one file per operator
(`aggregation_paths` in the config) or one file with a column family per
operator (nnU-Net's `f*_mean` / `f*_median` / `f*_max`). With neither, the
script says so and reports the single configured variant rather than printing
three identical rows.
