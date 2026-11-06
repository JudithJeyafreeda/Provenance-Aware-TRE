#!/usr/bin/env python3

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd


def as_bool(x):
    return str(x).strip().lower() in ("true", "1", "1.0", "yes")


def base_id(e):
    return re.sub(r"_(remove_evidence|mask_temporal_cue|silent_removal|silent_keep)$", "", str(e))


def auroc(pos, neg):
    """P(score_clean > score_control), ties count 1/2 (Mann-Whitney)."""
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank().values
    r = ranks[:len(pos)].sum()
    return (r - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def load_meta(D):
    notes = pd.read_csv(os.path.join(D, "synthetic_notes_v1.csv"))
    gold = dict(zip(notes["note_id"].astype(str), notes["gold_relation"]))
    m = pd.read_csv(os.path.join(D, "evidence_removed_only.csv"))
    notice = set()
    if "masked_cue" in m:
        mm = m[m["variant"] == "mask_temporal_cue"]
        # masked_cue is True when a cue phrase was replaced by the mask token,
        # False when no cue phrase was found and the notice sentence was appended
        tok = mm["masked_cue"].map(as_bool)
        notice = set(mm.loc[~tok, "note_id"].astype(str))
    sc = os.path.join(D, "silent_controls")
    r = pd.read_csv(os.path.join(sc, "silent_removal_controls.csv"))
    tier = dict(zip(r["note_id"].astype(str), r["validity_tier"]))
    qp = os.path.join(sc, "silent_removal_controls_qwen.csv")
    tier_q = {}
    if os.path.exists(qp):
        q = pd.read_csv(qp)
        tier_q = dict(zip(q["note_id"].astype(str), q["validity_tier"]))
    return gold, notice, tier, tier_q


def load(path, gold, score_col):
    df = pd.read_csv(path)
    df["example_id"] = df["example_id"].astype(str)
    df["base"] = df["example_id"].map(base_id)
    df["gold"] = df["base"].map(gold)
    df["abst"] = df["abstained"].map(as_bool)
    df["ok"] = (~df["abst"]) & (df["answer"].astype(str) == df["gold"].astype(str))
    if score_col in df:
        s = pd.to_numeric(df[score_col], errors="coerce")
        if score_col == "llm_confidence" and "raw_relation" in df:
            s = s.where(df["raw_relation"].notna(), -1.0)
        df["score"] = s.fillna(-1.0)
    else:
        df["score"] = np.nan
    return df


def metrics(name, ctrl, clean):
    n = len(ctrl)
    a = int((~ctrl["abst"]).sum())
    c = int(ctrl["ok"].sum())
    pc = clean[clean["example_id"].isin(set(ctrl["base"]))]
    row = {"set": name, "n": n,
           "coverage": a / n if n else np.nan,
           "abstention": 1 - a / n if n else np.nan,
           "accuracy": c / a if a else np.nan,
           "clean_cov_paired": (~pc["abst"]).mean() if len(pc) else np.nan,
           "clean_acc_paired": (pc["ok"].sum() / (~pc["abst"]).sum()) if (~pc["abst"]).sum() else np.nan}
    if name != "clean":
        row["auroc"] = auroc(clean["score"], ctrl["score"]) if clean["score"].notna().all() else np.nan
    return row


def score_run(files, gold, notice, tier, tier_q, score_col):
    clean = load(files["clean"], gold, score_col)
    rows = [metrics("clean", clean, clean)]
    if files.get("marker"):
        mk = load(files["marker"], gold, score_col)
        rows.append(metrics("marker", mk, clean))
        rows.append(metrics("remove_evidence", mk[mk["example_id"].str.endswith("_remove_evidence")], clean))
        mt = mk[mk["example_id"].str.endswith("_mask_temporal_cue")]
        if notice:
            rows.append(metrics("mask_token", mt[~mt["example_id"].isin(notice)], clean))
            rows.append(metrics("notice_appended", mt[mt["example_id"].isin(notice)], clean))
    if files.get("removal"):
        rm = load(files["removal"], gold, score_col)
        rm["tier"] = rm["example_id"].map(tier)
        rm["tier_q"] = rm["example_id"].map(tier_q)
        rows.append(metrics("silent_strict", rm[rm["tier"] == "strict"], clean))
        if tier_q:
            rows.append(metrics("silent_both_strict", rm[(rm["tier"] == "strict") & (rm["tier_q"] == "strict")], clean))
        rows.append(metrics("silent_implied", rm[rm["tier"] == "implied_only"], clean))
        rows.append(metrics("silent_all", rm, clean))
    if files.get("keep"):
        rows.append(metrics("silent_keep", load(files["keep"], gold, score_col), clean))
    return rows


def first(d, pat):
    h = sorted(glob.glob(os.path.join(d, pat)))
    return h[0] if h else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--runs_dir", default=None)
    ap.add_argument("--llm", action="append", default=[], help="name=folder with llm_availability_gated_*_predictions.csv")
    ap.add_argument("--out_prefix", default="gate_variants_summary")
    a = ap.parse_args()

    gold, notice, tier, tier_q = load_meta(a.data_dir)
    allrows = []

    if a.runs_dir:
        for run in sorted(glob.glob(os.path.join(a.runs_dir, "*_seed*"))):
            m = re.match(r"(.+)_(frozen|unfrozen)_seed(\d+)$", os.path.basename(run))
            if not m:
                continue
            files = {"clean": first(run, "*_clean_test_predictions.csv"),
                     "marker": first(run, "*_unavailable_test_predictions.csv"),
                     "removal": first(run, "*_silent_removal_controls*_predictions.csv"),
                     "keep": first(run, "*_silent_keep_controls*_predictions.csv")}
            if not files["clean"]:
                continue
            for r in score_run(files, gold, notice, tier, tier_q, "availability_probability"):
                allrows.append({"gate": "BERT-" + m.group(1), "encoder": m.group(2),
                                "seed": int(m.group(3)), **r})

    for spec in a.llm:
        name, d = spec.split("=", 1)
        files = {"clean": first(d, "llm_availability_gated_synthetic_notes_relation_only_predictions.csv"),
                 "marker": first(d, "llm_availability_gated_evidence_removed_only_predictions.csv"),
                 "removal": first(d, "llm_availability_gated_silent_removal_controls_predictions.csv"),
                 "keep": first(d, "llm_availability_gated_silent_keep_controls_predictions.csv")}
        if not files["clean"]:
            print(f"skip {name}: no clean predictions in {d}")
            continue
        for r in score_run(files, gold, notice, tier, tier_q, "llm_confidence"):
            allrows.append({"gate": "LLM-" + name, "encoder": "-", "seed": 0, **r})

    if not allrows:
        raise SystemExit("nothing to summarise")
    runs = pd.DataFrame(allrows)
    runs.to_csv(f"{a.out_prefix}_runs.csv", index=False)

    vals = ["n", "coverage", "abstention", "accuracy", "clean_cov_paired", "clean_acc_paired", "auroc"]
    g = runs.groupby(["gate", "encoder", "set"], sort=False)
    agg = g[vals].mean().add_suffix("_mean").join(g[vals].std(ddof=1).add_suffix("_sd"))
    agg["seeds"] = g["seed"].nunique()
    agg = agg.reset_index()
    agg.to_csv(f"{a.out_prefix}_agg.csv", index=False)

    def cell(gate, enc, s, col, pct=True):
        r = agg[(agg.gate == gate) & (agg.encoder == enc) & (agg.set == s)]
        if r.empty or pd.isna(r[f"{col}_mean"].iloc[0]):
            return "--"
        m, sd = r[f"{col}_mean"].iloc[0], r[f"{col}_sd"].iloc[0]
        f = (lambda x: f"{100 * x:.1f}") if pct else (lambda x: f"{x:.3f}")
        return f(m) + ("" if pd.isna(sd) else f" ± {f(sd)}")

    hdr = ["Gate", "Enc.", "Seeds", "Clean cov.", "Clean acc.", "Abst. marker",
           "Abst. strict", "Abst. both-strict", "Abst. implied", "Abst. keep",
           "Acc. strict", "AUROC marker", "AUROC strict", "AUROC keep"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for (gate, enc), _ in agg.groupby(["gate", "encoder"], sort=False):
        sd = agg[(agg.gate == gate) & (agg.encoder == enc)]["seeds"].iloc[0]
        lines.append("| " + " | ".join([
            gate, enc, str(sd),
            cell(gate, enc, "clean", "coverage"), cell(gate, enc, "clean", "accuracy"),
            cell(gate, enc, "marker", "abstention"), cell(gate, enc, "silent_strict", "abstention"),
            cell(gate, enc, "silent_both_strict", "abstention"), cell(gate, enc, "silent_implied", "abstention"),
            cell(gate, enc, "silent_keep", "abstention"), cell(gate, enc, "silent_strict", "accuracy"),
            cell(gate, enc, "marker", "auroc", False), cell(gate, enc, "silent_strict", "auroc", False),
            cell(gate, enc, "silent_keep", "auroc", False)]) + " |")
    md = "\n".join(lines)
    with open(f"{a.out_prefix}.md", "w") as f:
        f.write(md + "\n\nPercentages are means ± SD over seeds; AUROC separates clean inputs from each set "
                "(0.5 = none). A good evidence gate abstains on marker, strict and both-strict "
                "but not on keep, and has high AUROC on strict with AUROC near 0.5 on keep.\n")
    print(md)
    print(f"\nwrote {a.out_prefix}_runs.csv, {a.out_prefix}_agg.csv, {a.out_prefix}.md")


if __name__ == "__main__":
    main()
