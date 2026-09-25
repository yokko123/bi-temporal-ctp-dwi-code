#!/usr/bin/env python
"""Split balanced_features.h5 into LVO / non-LVO and mRS favorable / unfavorable subsets.

Single streaming pass: every row belongs to exactly one LVO group and exactly one mRS
group, so all four outputs are written together (~50 GB read, ~101 GB written) rather
than re-reading the source once per output.

Labels come from 3d_glcm/updated_csv/glcm_slice_with_clinical.csv (`cohort`, `mrs_group`),
which is the same source the paper's cohort counts come from: LVO 63 / non-LVO 46,
favorable 79 / unfavorable 30 over the full 109 patients.
"""
import os, re, sys, time
import numpy as np
import pandas as pd
import h5py

BASE = os.environ.get("FEATURE_WORK_ROOT", "/path/to/feature-work")
SRC = f"{BASE}/mJ-Net_pytorch/balanced_features.h5"
CLIN = f"{BASE}/3d_glcm/updated_csv/glcm_slice_with_clinical.csv"
OUTDIR = f"{BASE}/mJ-Net_pytorch"
ROWS_PER_BLOCK = 200_000

SPLITS = {
    "lvo":       ("cohort",    "LVO"),
    "non_lvo":   ("cohort",    "Non-LVO"),
    "mrs_fav":   ("mrs_group", "Favorable (0-2)"),
    "mrs_unfav": ("mrs_group", "Unfavorable (3-6)"),
}


def to_pid(s):
    m = re.match(r"sub-stroke_(\d+)_(\d+)$", s)
    return int(m.group(1)) * 1000 + int(m.group(2))


def main():
    clin = (pd.read_csv(CLIN, usecols=["patient_id", "cohort", "mrs_group"])
              .drop_duplicates("patient_id"))
    clin["pid"] = clin["patient_id"].map(to_pid)
    assert clin["pid"].is_unique, "duplicate pid in clinical table"

    with h5py.File(SRC, "r") as f:
        pid = f["pid"][:]
        n = len(pid)
        assert f["X"].shape[0] == n and f["y"].shape[0] == n, "ragged source datasets"

    h5_pids = np.unique(pid)
    known = set(clin["pid"])
    missing = sorted(set(h5_pids.tolist()) - known)
    if missing:
        sys.exit(f"ERROR: {len(missing)} h5 patients have no clinical label: {missing}")
    print(f"[  ] {n:,} rows, {len(h5_pids)} patients, all labelled")

    # row -> group, via a pid-indexed lookup
    lut = {p: (c, m) for p, c, m in
           zip(clin["pid"], clin["cohort"], clin["mrs_group"])}
    masks, counts = {}, {}
    for name, (col, val) in SPLITS.items():
        want = {p for p, (c, m) in lut.items()
                if (c if col == "cohort" else m) == val}
        pats = sorted(want & set(h5_pids.tolist()))
        masks[name] = np.isin(pid, np.array(pats, dtype=pid.dtype))
        counts[name] = (int(masks[name].sum()), len(pats))
        print(f"[  ] {name:10s} {counts[name][0]:12,} rows  {counts[name][1]:3d} patients")

    # every row must land in exactly one group of each pair
    assert (masks["lvo"] ^ masks["non_lvo"]).all(), "LVO split is not a partition"
    assert (masks["mrs_fav"] ^ masks["mrs_unfav"]).all(), "mRS split is not a partition"

    with h5py.File(SRC, "r") as f:
        Xs, ys = f["X"], f["y"]
        outs, dsets, offs = {}, {}, {}
        for name, (nrows, npat) in counts.items():
            path = f"{OUTDIR}/balanced_features_{name}.h5"
            if os.path.exists(path):
                sys.exit(f"ERROR: refusing to overwrite existing {path}")
            o = h5py.File(path, "w")
            outs[name] = o
            dsets[name] = {
                "X":   o.create_dataset("X", (nrows, 256), dtype="float32",
                                        chunks=(10000, 256)),
                "pid": o.create_dataset("pid", (nrows,), dtype="int32", chunks=(10000,)),
                "y":   o.create_dataset("y", (nrows,), dtype="int32", chunks=(10000,)),
            }
            col, val = SPLITS[name]
            o.attrs.update(source=os.path.basename(SRC), label_column=col,
                           label_value=val, n_patients=npat, n_rows=nrows,
                           label_source=os.path.relpath(CLIN, BASE),
                           created=time.strftime("%Y-%m-%d %H:%M:%S"))
            offs[name] = 0
            print(f"[  ] -> {path}")

        t0 = time.time()
        for s in range(0, n, ROWS_PER_BLOCK):
            e = min(s + ROWS_PER_BLOCK, n)
            Xb, yb, pb = Xs[s:e], ys[s:e], pid[s:e]
            for name in SPLITS:
                m = masks[name][s:e]
                k = int(m.sum())
                if not k:
                    continue
                o, d = offs[name], dsets[name]
                d["X"][o:o + k] = Xb[m]
                d["pid"][o:o + k] = pb[m]
                d["y"][o:o + k] = yb[m]
                offs[name] = o + k
            if (s // ROWS_PER_BLOCK) % 25 == 0 or e == n:
                el = time.time() - t0
                pct = e / n
                print(f"[{pct*100:5.1f}%] {e:,}/{n:,} rows  {el/60:.1f} min elapsed"
                      f"  eta {(el/max(pct,1e-9)-el)/60:.1f} min", flush=True)

        for name in SPLITS:
            assert offs[name] == counts[name][0], \
                f"{name}: wrote {offs[name]} of {counts[name][0]} rows"
            outs[name].close()

    print(f"\n[ok] done in {(time.time()-t0)/60:.1f} min")
    for name in SPLITS:
        p = f"{OUTDIR}/balanced_features_{name}.h5"
        print(f"     {os.path.basename(p):36s} {os.path.getsize(p)/1e9:7.1f} GB")


if __name__ == "__main__":
    main()
