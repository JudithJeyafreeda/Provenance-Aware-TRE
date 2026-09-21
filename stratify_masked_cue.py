#!/usr/bin/env python3
"""Stratify abstention results on the evidence-unavailable controls.

No new model inference is needed. The script joins an existing control
prediction file to the control CSV (evidence_removed_only.csv) on example_id
and reports, for each model:

  * remove_evidence: number of distinct input texts (1 means a single input)
  * mask_temporal_cue split into
      masked_cue = True   a cue was found and replaced by [MASKED_TEMPORAL_CUE]
      masked_cue = False  no cue was found; the note is unchanged and an
                          omission sentence was appended
  * any other variant in the control CSV (for example silent_removal), reported as
    its own stratum. For silent_keep (evidence available) read coverage and
    accuracy, not abstention recall: the gate should answer there.
  * the same metrics as evaluate_control_abstain_v1.py, with Wilson 95% CIs

Usage:
  python stratify_masked_cue.py \
    --control_csv evidence_removed_only.csv \
    --control_predictions llm_availability_gated_control_predictions_eval.csv \
    --clean_predictions llm_availability_gated_clean_predictions_eval.csv \
    --out_csv stratified_mask_results.csv

--clean_predictions is optional. When given, each stratum also reports the
clean-input coverage and answered accuracy of the SAME base notes, so a note
answered when clean but withheld once an omission notice is appended shows up
directly (paired comparison).
"""
import argparse
import math

import numpy as np
import pandas as pd


def parse_bool(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return bool(v) if not pd.isna(v) else False
    return str(v).strip().lower() in {"1", "true", "yes", "y", "abstain", "abstained"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def summarize(df):
    n = len(df)
    answered = ~df["abstained_bool"]
    a = int(answered.sum())
    correct = df["correct_num"].fillna(0).eq(1) & answered
    ac = int(correct.sum())
    abst = n - a
    unsafe = a - ac
    lo, hi = wilson(abst, n)
    ulo, uhi = wilson(unsafe, n)
    return {
        "n": n,
        "n_answered": a,
        "abstention_recall": abst / n if n else float("nan"),
        "abst_ci_low": lo,
        "abst_ci_high": hi,
        "coverage_on_unavailable": a / n if n else float("nan"),
        "unsafe_answer_rate": unsafe / n if n else float("nan"),
        "unsafe_ci_low": ulo,
        "unsafe_ci_high": uhi,
        "answered_accuracy": ac / a if a else float("nan"),
    }


def paired_clean(sub, clean):
    """Clean coverage / accuracy on the base notes of a control stratum."""
    c = clean[clean["example_id"].isin(set(sub["base_id"]))]
    if not len(c):
        return {"clean_n_same_notes": 0}
    ans = ~c["abstained_bool"]
    a = int(ans.sum())
    ac = int((c["correct_num"].fillna(0).eq(1) & ans).sum())
    return {
        "clean_n_same_notes": len(c),
        "clean_coverage_same_notes": a / len(c),
        "clean_answered_accuracy_same_notes": ac / a if a else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control_csv", required=True)
    ap.add_argument("--control_predictions", required=True)
    ap.add_argument("--clean_predictions", default=None)
    ap.add_argument("--out_csv", default="stratified_mask_results.csv")
    args = ap.parse_args()

    ctrl = pd.read_csv(args.control_csv)
    if "example_id" not in ctrl.columns:
        ctrl["example_id"] = ctrl["note_id"]
    ctrl["example_id"] = ctrl["example_id"].astype(str)

    print("== Control set composition (from control CSV) ==")
    for v, g in ctrl.groupby("variant"):
        print(f"{v}: n={len(g)}, distinct texts={g['text'].nunique()}")
    m = ctrl[ctrl["variant"] == "mask_temporal_cue"]
    tok = m["text"].str.contains("[MASKED_TEMPORAL_CUE]", regex=False)
    print(f"mask_temporal_cue: token inserted={int(tok.sum())}, "
          f"note unchanged + notice={int((~tok).sum())}")

    pred = pd.read_csv(args.control_predictions)
    pred["example_id"] = pred["example_id"].astype(str)
    pred["abstained_bool"] = pred["abstained"].map(parse_bool)
    pred["correct_num"] = pd.to_numeric(pred["correct"], errors="coerce")
    keep = ["example_id", "variant"]
    if "masked_cue" in ctrl.columns:
        keep.append("masked_cue")
    if "base_note_id" in ctrl.columns:
        keep.append("base_note_id")
    merged = pred.merge(ctrl[keep], on="example_id", how="left")
    unmatched = int(merged["variant"].isna().sum())
    if unmatched:
        print(f"WARNING: {unmatched} prediction rows did not match the control CSV")

    if "masked_cue" not in merged.columns:
        merged["masked_cue"] = np.nan
    if "base_note_id" in merged.columns:
        merged["base_id"] = merged["base_note_id"].astype(str)
    else:
        merged["base_id"] = merged["example_id"].str.replace(
            r"_(remove_evidence|mask_temporal_cue)$", "", regex=True)
    clean = None
    if args.clean_predictions:
        clean = pd.read_csv(args.clean_predictions)
        clean["example_id"] = clean["example_id"].astype(str)
        clean["abstained_bool"] = clean["abstained"].map(parse_bool)
        clean["correct_num"] = pd.to_numeric(clean["correct"], errors="coerce")
    def extra(sub):
        return paired_clean(sub, clean) if clean is not None else {}
    rows = []
    for model, gm in merged.groupby("model"):
        rem = gm[gm["variant"] == "remove_evidence"]
        if len(rem):
            rows.append({"model": model, "stratum": "remove_evidence", **summarize(rem), **extra(rem)})
        msk = gm[gm["variant"] == "mask_temporal_cue"].copy()
        if len(msk):
            rows.append({"model": model, "stratum": "mask_temporal_cue (all)", **summarize(msk), **extra(msk)})
            msk["mc"] = msk["masked_cue"].map(parse_bool)
            for flag, label in [(True, "mask_temporal_cue: token inserted (masked_cue=True)"),
                                (False, "mask_temporal_cue: note intact + notice (masked_cue=False)")]:
                sub = msk[msk["mc"] == flag]
                if len(sub):
                    rows.append({"model": model, "stratum": label, **summarize(sub), **extra(sub)})
        # any other control variant (for example silent_removal, silent_keep)
        for v in sorted(x for x in gm["variant"].dropna().unique()
                        if x not in ("remove_evidence", "mask_temporal_cue")):
            sub = gm[gm["variant"] == v]
            rows.append({"model": model, "stratum": v, **summarize(sub), **extra(sub)})
    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", "{:.3f}".format):
        print("\n== Stratified results ==")
        print(out.to_string(index=False))
    print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
