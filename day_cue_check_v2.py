#!/usr/bin/env python3

import argparse
import glob
import os
import re

import pandas as pd

DAY = re.compile(r"\bday\s*(\d+)", re.I)
DIGIT = re.compile(r"\d")


def as_bool(x):
    return str(x).strip().lower() in ("true", "1", "1.0", "yes")


def texts(D):
    t = {}
    notes = pd.read_csv(os.path.join(D, "synthetic_notes_v1.csv"))
    t.update(zip(notes["note_id"].astype(str), notes["text"].astype(str)))
    m = pd.read_csv(os.path.join(D, "evidence_removed_only.csv"))
    t.update(zip(m["note_id"].astype(str), m["text"].astype(str)))
    notice = set()
    if "masked_cue" in m:
        mm = m[m["variant"] == "mask_temporal_cue"]
        # masked_cue is True when a cue phrase was replaced by the mask token,
        # False when no cue phrase was found and the notice sentence was appended
        tok = mm["masked_cue"].map(as_bool)
        notice = set(mm.loc[~tok, "note_id"].astype(str))
    sc = os.path.join(D, "silent_controls")
    r = pd.read_csv(os.path.join(sc, "silent_removal_controls.csv"))
    t.update(zip(r["note_id"].astype(str), r["text"].astype(str)))
    tier = dict(zip(r["note_id"].astype(str), r["validity_tier"]))
    k = pd.read_csv(os.path.join(sc, "silent_keep_controls.csv"))
    t.update(zip(k["note_id"].astype(str), k["text"].astype(str)))
    return t, tier, notice


def label_set(eid, tier, notice):
    if eid.endswith("_remove_evidence"):
        return "remove_evidence"
    if eid.endswith("_mask_temporal_cue"):
        return "notice_appended" if eid in notice else "mask_token"
    if eid.endswith("_silent_removal"):
        return "silent_" + str(tier.get(eid, "?"))
    if eid.endswith("_silent_keep"):
        return "silent_keep"
    return "clean"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--runs_dir", required=True)
    ap.add_argument("--out_csv", default="day_cue_check.csv")
    a = ap.parse_args()
    txt, tier, notice = texts(a.data_dir)

    rows = []
    for run in sorted(glob.glob(os.path.join(a.runs_dir, "*_seed*"))):
        m = re.match(r"(.+)_(frozen|unfrozen)_seed(\d+)$", os.path.basename(run))
        if not m:
            continue
        dfs = [pd.read_csv(f) for f in glob.glob(os.path.join(run, "*_test_predictions.csv"))]
        if not dfs:
            continue
        df = pd.concat(dfs, ignore_index=True)
        df["example_id"] = df["example_id"].astype(str)
        df["text"] = df["example_id"].map(txt)
        df["n_day"] = df["text"].fillna("").map(lambda s: len(DAY.findall(s)))
        df["gate_abst"] = df["abstained"].map(as_bool)
        df["rule_abst"] = df["n_day"] == 0
        df["no_digit"] = ~df["text"].fillna("").str.contains(DIGIT)
        df["set"] = df["example_id"].map(lambda e: label_set(e, tier, notice))
        for s, g in df.groupby("set"):
            nd = g[g["n_day"] == 0]
            wd = g[g["n_day"] > 0]
            rows.append({"condition": m.group(1), "encoder": m.group(2), "seed": int(m.group(3)),
                         "set": s, "n": len(g),
                         "share_no_day": (g["n_day"] == 0).mean(),
                         "gate_abst": g["gate_abst"].mean(),
                         "rule_abst": g["rule_abst"].mean(),
                         "agree_with_rule": (g["gate_abst"] == g["rule_abst"]).mean(),
                         "gate_abst_if_no_day": nd["gate_abst"].mean() if len(nd) else float("nan"),
                         "gate_abst_if_day": wd["gate_abst"].mean() if len(wd) else float("nan"),
                         "share_no_digit": g["no_digit"].mean(),
                         "agree_with_no_digit": (g["gate_abst"] == g["no_digit"]).mean()})
    if not rows:
        raise SystemExit("no runs found")
    r = pd.DataFrame(rows)
    r.to_csv(a.out_csv, index=False)

    agg = (r.groupby(["condition", "encoder", "set"])
             [["n", "share_no_day", "gate_abst", "rule_abst", "agree_with_rule",
               "gate_abst_if_no_day", "gate_abst_if_day", "share_no_digit",
               "agree_with_no_digit"]].mean().round(3))
    order = ["clean", "remove_evidence", "mask_token", "notice_appended",
             "silent_strict", "silent_implied_only", "silent_keep"]
    agg = agg.reset_index()
    agg["set"] = pd.Categorical(agg["set"], [s for s in order if s in set(agg["set"])] +
                                [s for s in agg["set"].unique() if s not in order])
    agg = agg.sort_values(["condition", "encoder", "set"])
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 500)
    print(agg.to_string(index=False))
    # one-line verdict per gate: overall agreement with the day-number rule
    tot = r.groupby(["condition", "encoder"]).apply(
        lambda g: (g["agree_with_rule"] * g["n"]).sum() / g["n"].sum()).round(3)
    tot2 = r.groupby(["condition", "encoder"]).apply(
        lambda g: (g["agree_with_no_digit"] * g["n"]).sum() / g["n"].sum()).round(3)
    print("\nOverall agreement with the rules (all test inputs):")
    print(pd.DataFrame({"no 'day N' -> abstain": tot, "no digit at all -> abstain": tot2}).to_string())
    print(f"\nwrote {a.out_csv}")


if __name__ == "__main__":
    main()
