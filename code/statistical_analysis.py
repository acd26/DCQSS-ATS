"""Group bootstrap intervals and paired exact McNemar tests."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest


def metrics(y, p):
    y, p = np.asarray(y, bool), np.asarray(p, bool)
    tp, tn, fp, fn = [int(v.sum()) for v in [y & p, ~y & ~p, ~y & p, y & ~p]]
    return dict(
        n=len(y),
        TP=tp,
        TN=tn,
        FP=fp,
        FN=fn,
        precision=tp / (tp + fp) if tp + fp else 0.0,
        recall=tp / (tp + fn) if tp + fn else 0.0,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        accuracy=(tp + tn) / len(y) if len(y) else 0.0,
    )


def analyze(decisions, groups, replicates=2000, seed=777):
    if replicates < 1:
        raise ValueError("replicates must be positive")
    df = decisions.copy()
    if "status" in df and df.status.ne("ok").any():
        raise ValueError(
            "Failed detections must be resolved before statistical analysis"
        )
    if df.pred.dtype != bool:
        v = df.pred.astype(str).str.lower()
        assert v.isin(["true", "false"]).all()
        df["pred"] = v.eq("true")
    assert not df.duplicated(["dataset", "label", "stem", "variant"]).any()
    df = df.merge(
        groups[["dataset", "label", "stem", "group"]],
        on=["dataset", "label", "stem"],
        validate="many_to_one",
    )
    assert len(df) == len(decisions), "Missing recording group identifiers"
    summaries = [
        dict(dataset=ds, variant=variant, **metrics(g.label.eq("tampered"), g.pred))
        for (ds, variant), g in df.groupby(["dataset", "variant"])
    ]
    rng = np.random.default_rng(seed)
    intervals = []
    for ds, g in df[df.variant.eq("dcqss")].groupby("dataset"):
        ids = g.group.astype(str).to_numpy()
        groups_ix = [np.flatnonzero(ids == v) for v in sorted(set(ids))]
        y, p = g.label.eq("tampered").to_numpy(), g.pred.to_numpy()
        values = {k: [] for k in ["precision", "recall", "f1", "accuracy"]}
        for _ in range(replicates):
            ix = np.concatenate(
                [groups_ix[i] for i in rng.integers(0, len(groups_ix), len(groups_ix))]
            )
            m = metrics(y[ix], p[ix])
            for key in values:
                values[key].append(m[key])
        m = metrics(y, p)
        for key, v in values.items():
            lo, hi = np.quantile(v, [0.025, 0.975])
            intervals.append(
                dict(
                    dataset=ds,
                    metric=key,
                    value=m[key],
                    ci_low=lo,
                    ci_high=hi,
                    replicates=replicates,
                    unit="authentic/forged source pair",
                )
            )
    paired = []
    for ds, g in df.groupby("dataset"):
        pivot = g.pivot(index=["label", "stem"], columns="variant", values="pred")
        if not {"cqss", "diff", "dcqss"} <= set(pivot):
            continue
        if pivot.isna().any().any():
            raise ValueError("Unpaired feature decisions")
        truth = np.array([label == "tampered" for label, _ in pivot.index])
        final = pivot.dcqss.to_numpy(bool) == truth
        for variant in ["cqss", "diff"]:
            other = pivot[variant].to_numpy(bool) == truth
            wins = int((final & ~other).sum())
            losses = int((~final & other).sum())
            paired.append(
                dict(
                    dataset=ds,
                    comparison="dcqss_vs_" + variant,
                    wins=wins,
                    losses=losses,
                    mcnemar_exact_p=(
                        binomtest(wins, wins + losses).pvalue if wins + losses else 1.0
                    ),
                )
            )
    return pd.DataFrame(summaries), pd.DataFrame(intervals), pd.DataFrame(paired)


def filename_groups(frame):
    """Pair released authentic/forged recordings by their filename identifier."""
    groups = frame[["dataset", "label", "stem"]].drop_duplicates().copy()
    groups["group"] = groups.stem.astype(str).str.rsplit("_", n=1).str[-1]
    if groups.duplicated(["dataset", "label", "group"]).any():
        raise ValueError("Ambiguous filename pairing")
    sizes = groups.groupby(["dataset", "group"]).label.nunique()
    if not sizes.eq(2).all():
        raise ValueError("Each source identifier needs authentic and forged recordings")
    return groups


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--decisions", type=Path, default=Path("outputs/decisions.csv"))
    p.add_argument("--out", type=Path, default=Path("outputs/statistics"))
    a = p.parse_args()
    frame = pd.read_csv(a.decisions, dtype={"stem": str})
    groups = filename_groups(frame)
    results = analyze(frame, groups, replicates=2000, seed=777)
    a.out.mkdir(parents=True, exist_ok=True)
    for name, frame in zip(
        ["feature_summary.csv", "bootstrap.csv", "paired_significance.csv"], results
    ):
        frame.to_csv(a.out / name, index=False)
    print(a.out)


if __name__ == "__main__":
    main()
