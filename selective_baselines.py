#!/usr/bin/env python3
"""
"""
import argparse

import numpy as np
import pandas as pd


def auroc(pos, neg):
    """P(score_pos > score_neg) + 0.5 P(tie), via ranks (Mann-Whitney)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank(method="average").to_numpy()
    r_pos = ranks[: len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def boot_ci(pos, neg, n_boot, seed):
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    vals = [auroc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(n_boot)]
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def threshold_for_coverage(scores, target):
    """Smallest threshold t such that mean(scores >= t) <= target, preferring the
    achieved coverage closest to the target."""
    s = np.sort(np.asarray(scores, float))[::-1]
    finite = s[np.isfinite(s)]
    if len(finite) == 0:
        return np.inf
    candidates = np.unique(finite)
    best, best_gap = np.inf, np.inf
    for t in candidates:
        cov = np.mean(s >= t)
        gap = abs(cov - target)
        if gap < best_gap - 1e-12:
            best, best_gap = t, gap
    return best


def aurc(scores, correct):
    order = np.argsort(-np.asarray(scores, float), kind="mergesort")
    c = np.asarray(correct, float)[order]
    risks = np.cumsum(1 - c) / np.arange(1, len(c) + 1)
    return float(risks.mean())


SUFFIXES = ("_remove_evidence", "_mask_temporal_cue", "_silent_removal", "_silent_keep")


def base_id(x):
    for suf in SUFFIXES:
        if x.endswith(suf):
            return x[: -len(suf)]
    return x


def load(path, gold, pred_col, score_col):
    df = pd.read_csv(path)
    df["example_id"] = df["example_id"].astype(str)
    missing = {pred_col, score_col} - set(df.columns)
    if missing:
        raise ValueError(f"{path} lacks columns {sorted(missing)}: export them first (see patches)")
    g = gold.set_index("example_id")
    # control rows inherit the gold relation of their base note when not listed themselves
    df["_gold_key"] = [e if e in g.index else base_id(e) for e in df["example_id"]]
    extra = [c for c in g.columns]
    df = df.join(g[extra], on="_gold_key")
    if df["gold_relation"].isna().any():
        n = int(df["gold_relation"].isna().sum())
        raise ValueError(f"{path}: {n} rows have no gold relation in the --gold files")
    pred = df[pred_col]
    score = pd.to_numeric(df[score_col], errors="coerce")
    has_pred = pred.notna() & pred.astype(str).str.len().gt(0) & pred.astype(str).ne("None")
    df["_score"] = np.where(has_pred & score.notna(), score, -np.inf)
    df["_correct"] = (has_pred & (pred.astype(str) == df["gold_relation"].astype(str))).astype(int)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clean", required=True)
    p.add_argument("--condition", action="append", required=True, help="name=path; repeatable")
    p.add_argument("--gold", nargs="+", required=True,
                   help="CSV files with note_id and gold_relation (and optional tier column)")
    p.add_argument("--pred_col", required=True)
    p.add_argument("--score_col", required=True)
    p.add_argument("--target_coverage", type=float, required=True)
    p.add_argument("--tier_col", default=None)
    p.add_argument("--tier", default=None)
    p.add_argument("--tier_condition", action="append", default=[],
                   help="condition name(s) the tier filter applies to; repeatable")
    p.add_argument("--label", default=None)
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    cols = ["note_id", "gold_relation"] + ([a.tier_col] if a.tier_col else [])
    g = pd.concat([pd.read_csv(f).reindex(columns=cols) for f in a.gold], ignore_index=True)
    g = g.dropna(subset=["gold_relation"]).drop_duplicates("note_id")
    g = g.rename(columns={"note_id": "example_id"})
    g["example_id"] = g["example_id"].astype(str)

    clean = load(a.clean, g, a.pred_col, a.score_col)
    t = threshold_for_coverage(clean["_score"], a.target_coverage)
    clean_ans = clean["_score"] >= t
    rows = [{
        "model_score": a.label or f"{a.pred_col}/{a.score_col}", "condition": "clean", "n": len(clean),
        "threshold": t, "coverage": clean_ans.mean(), "abstention": 1 - clean_ans.mean(),
        "answered_accuracy": clean.loc[clean_ans, "_correct"].mean() if clean_ans.any() else np.nan,
        "auroc_vs_clean": np.nan, "auroc_lo": np.nan, "auroc_hi": np.nan,
        "aurc_clean": aurc(clean["_score"], clean["_correct"]),
    }]
    for spec in a.condition:
        name, path = spec.split("=", 1)
        d = load(path, g, a.pred_col, a.score_col)
        if a.tier_col and a.tier and name in a.tier_condition:
            d = d[d[a.tier_col].eq(a.tier)]
        ans = d["_score"] >= t
        auc = auroc(clean["_score"].replace(-np.inf, -1e9), d["_score"].replace(-np.inf, -1e9))
        lo, hi = boot_ci(clean["_score"].replace(-np.inf, -1e9), d["_score"].replace(-np.inf, -1e9),
                         a.n_boot, a.seed)
        rows.append({
            "model_score": a.label or f"{a.pred_col}/{a.score_col}", "condition": name, "n": len(d),
            "threshold": t, "coverage": ans.mean(), "abstention": 1 - ans.mean(),
            "answered_accuracy": d.loc[ans, "_correct"].mean() if ans.any() else np.nan,
            "auroc_vs_clean": auc, "auroc_lo": lo, "auroc_hi": hi, "aurc_clean": np.nan,
        })
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
