#!/usr/bin/env python
"""Merge sharded navtest score CSVs into one PDMS, and compare against the paper.

run_pdm_score_cot appends an 'average' row to each CSV before writing, and its
logged "final average score" is computed AFTER that append, so it includes the
average as a data point. We drop those rows and average the real per-token
scores instead.
"""
import pathlib, sys
import pandas as pd

REPORTED = 89.0   # the checkpoint is literally AutoVLA_PDMS_89.ckpt

def main() -> int:
    root = pathlib.Path(sys.argv[1])
    csvs = sorted(root.glob("*/exp/**/*.csv"))
    if not csvs:
        print(f"no shard CSVs under {root}")
        return 1
    frames = []
    for c in csvs:
        # token must stay a string: ids like 3309408516525e17 are valid floats,
        # and pandas will happily turn them into 3.3e+29 and break the join.
        df = pd.read_csv(c, dtype={"token": str})
        df = df[df["token"] != "average"]
        frames.append(df)
        print(f"  {c.relative_to(root).parts[0]:>8s}: {len(df):5d} tokens from {c.name}")
    allrows = pd.concat(frames, ignore_index=True)
    allrows = allrows.drop_duplicates(subset="token", keep="first")

    n = len(allrows)
    valid = allrows["valid"].astype(bool)
    print(f"\ntokens: {n}   valid: {valid.sum()}   failed: {n - valid.sum()}")
    if "score" not in allrows.columns:
        print("no 'score' column -- every scenario failed; see shard*/pdm_score.log")
        return 1

    pdms = allrows.loc[valid, "score"].mean() * 100.0
    print(f"\n  PDMS (mean over {int(valid.sum())} valid tokens): {pdms:.2f}")
    print(f"  reported in paper / checkpoint name:              {REPORTED:.2f}")
    print(f"  difference:                                       {pdms - REPORTED:+.2f}")

    subs = [c for c in allrows.columns
            if c not in ("token", "valid", "score") and allrows[c].dtype.kind == "f"]
    if subs:
        print("\n  subscores (mean over valid):")
        for c in sorted(subs):
            print(f"    {c:<34s} {allrows.loc[valid, c].mean():.4f}")
    allrows.to_csv(root / "navtest_merged.csv", index=False)
    expected = 12146
    if n != expected:
        print(f"\n  WARNING: {expected - n} of {expected} navtest tokens are unscored.")
    print(f"\n  merged per-token rows -> {root/'navtest_merged.csv'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
