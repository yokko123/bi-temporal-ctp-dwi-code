# Synthetic demo cohort

No patient data ships with this repository, so stage 03 would otherwise be
unrunnable by anyone without access to the cohorts. `make_synthetic_cohort.py`
writes feature tables with the stage-02 schema for a made-up cohort.

```bash
python make_synthetic_cohort.py --out-dir data
cd ../03_analysis
python run_table3_region_pairs.py --config ../demo/data/config.yaml
```

The generator plants a single signal direction and shifts each ROI class along
it, ordered as the paper's finding: healthy contralateral brain at zero,
salvaged penumbra near it, the two core classes close together, infarcted
penumbra further out and infarcted non-hypoperfused brain furthest. Running
stage 03 on it should therefore give a significant Test 1, a non-significant
Test 2 and the largest separation on Test 4.

The numbers are not the paper's and mean nothing clinically. The point is that
the pipeline runs end to end and its output has the expected shape.

`data/` is gitignored.
