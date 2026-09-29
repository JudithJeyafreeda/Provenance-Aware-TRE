#!/usr/bin/env python3
"""Compute LLM evidence-span recall against regenerated (post-paraphrase) gold spans.


Usage:
  python compute_llm_evidence_recall.py \
    --predictions llm_v2_spans/llm_availability_gated_synthetic_notes_relation_only_predictions.csv \
    --gold_spans_v2 gold_spans_v2/gold_spans_v2.csv \
    --out_csv llm_v2_spans/llm_evidence_recall_v2.csv
"""
import argparse
import json

import pandas as pd


def char_overlap_both(text, span_a, span_b):
    """Returns (recall, precision, inter_len, len_a, len_b) where recall is the
    fraction of span_b covered by span_a and precision is the fraction of span_a
    that falls inside span_b -- both located as substrings of text. A predicted
    span that simply covers most of the note (e.g. everything except a fixed
    boilerplate opening/closing sentence) can score high recall against nearly
    any inner gold span without precisely localizing it; precision and the
    length ratio below are the check on that."""
    if not isinstance(span_b, str) or not span_b:
        return (None, None, None, None, None)
    b_start = text.find(span_b)
    if b_start < 0:
        return (None, None, None, None, None)
    b_end = b_start + len(span_b)
    if not isinstance(span_a, str) or not span_a:
        return (0.0, None, 0, 0, b_end - b_start)
    a_start = text.find(span_a)
    if a_start < 0:
        return (0.0, None, 0, len(span_a), b_end - b_start)
    a_end = a_start + len(span_a)
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    len_a = a_end - a_start
    len_b = b_end - b_start
    recall = inter / len_b if len_b else None
    precision = inter / len_a if len_a else None
    return (recall, precision, inter, len_a, len_b)


def char_overlap(text, span_a, span_b):
    """Backward-compatible wrapper returning recall only."""
    return char_overlap_both(text, span_a, span_b)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True,
                    help="LLM predictions CSV with predicted_evidence_span (from the patched v2 script)")
    ap.add_argument("--notes_csv", default=None,
                    help="notes CSV with note_id/text, if predictions lacks a usable text column; "
                         "usually not needed since gold_spans_v2.csv already carries text")
    ap.add_argument("--gold_spans_v2", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--exclude_abstained", action="store_true",
                    help="score only answered (non-abstained) rows, instead of scoring abstained rows as 0.0")
    args = ap.parse_args()

    pred = pd.read_csv(args.predictions)
    pred["example_id"] = pred["example_id"].astype(str)
    gold = pd.read_csv(args.gold_spans_v2)
    gold["note_id"] = gold["note_id"].astype(str)
    gold_acc = gold[gold["reason"] == "accepted"][
        ["note_id", "text", "gold_evidence_span_v2", "validity_tier"]
    ].rename(columns={"note_id": "example_id"})

    m = pred.merge(gold_acc, on="example_id", how="left")
    scorable = m["gold_evidence_span_v2"].notna()
    if args.exclude_abstained and "abstained" in m.columns:
        abstained_bool = m["abstained"].astype(str).str.lower().isin(["true", "1"])
        scorable = scorable & ~abstained_bool

    def row_metrics(r):
        return char_overlap_both(r["text"], r.get("predicted_evidence_span"), r["gold_evidence_span_v2"])

    m["evidence_recall_v2"] = None
    m["evidence_precision_v2"] = None
    m["predicted_span_len"] = None
    m["gold_span_len"] = None
    m["predicted_span_len_frac_of_note"] = None
    rows_metrics = m.loc[scorable].apply(row_metrics, axis=1)
    m.loc[scorable, "evidence_recall_v2"] = [v[0] for v in rows_metrics]
    m.loc[scorable, "evidence_precision_v2"] = [v[1] for v in rows_metrics]
    m.loc[scorable, "predicted_span_len"] = [v[3] for v in rows_metrics]
    m.loc[scorable, "gold_span_len"] = [v[4] for v in rows_metrics]
    note_len = m.loc[scorable, "text"].str.len()
    m.loc[scorable, "predicted_span_len_frac_of_note"] = (
        m.loc[scorable, "predicted_span_len"].astype(float) / note_len.replace(0, pd.NA)
    )
    rec = m.loc[scorable, "evidence_recall_v2"].astype(float)
    prec = m.loc[scorable, "evidence_precision_v2"].astype(float)
    denom = (rec + prec)
    m["evidence_f1_v2"] = None
    f1 = (2 * rec * prec / denom).where(denom > 0, 0.0)
    m.loc[scorable, "evidence_f1_v2"] = f1
    m.to_csv(args.out_csv, index=False)

    n = len(m)
    n_scorable = int(scorable.sum())
    vals = m.loc[scorable, "evidence_recall_v2"].dropna()
    prec_vals = m.loc[scorable, "evidence_precision_v2"].dropna()
    f1_vals = m.loc[scorable, "evidence_f1_v2"].dropna()
    len_frac_vals = m.loc[scorable, "predicted_span_len_frac_of_note"].dropna()
    report = {
        "n_rows": n,
        "n_scorable": n_scorable,
        "scorable_rate": round(n_scorable / n, 4) if n else None,
        "mean_evidence_recall_v2": round(float(vals.mean()), 4) if len(vals) else None,
        "median_evidence_recall_v2": round(float(vals.median()), 4) if len(vals) else None,
        "frac_zero_recall": round(float((vals == 0).mean()), 4) if len(vals) else None,
        "frac_full_1.0_recall": round(float((vals >= 0.999).mean()), 4) if len(vals) else None,
        "mean_evidence_precision_v2": round(float(prec_vals.mean()), 4) if len(prec_vals) else None,
        "median_evidence_precision_v2": round(float(prec_vals.median()), 4) if len(prec_vals) else None,
        "mean_evidence_f1_v2": round(float(f1_vals.mean()), 4) if len(f1_vals) else None,
        "mean_predicted_span_len_frac_of_note": round(float(len_frac_vals.mean()), 4) if len(len_frac_vals) else None,
        "median_predicted_span_len_frac_of_note": round(float(len_frac_vals.median()), 4) if len(len_frac_vals) else None,
        "frac_span_covers_over_80pct_of_note": round(float((len_frac_vals >= 0.8).mean()), 4) if len(len_frac_vals) else None,
    }
    if "gold_evidence_span_v2_tier" not in m.columns and "validity_tier" in m.columns:
        for tier, g in m[scorable].groupby("validity_tier"):
            v = g["evidence_recall_v2"].dropna()
            report[f"mean_recall__{tier}"] = round(float(v.mean()), 4) if len(v) else None

    if "abstained" in m.columns and not args.exclude_abstained:
        abstained_bool = m["abstained"].astype(str).str.lower().isin(["true", "1"])
        ans_scorable = scorable & ~abstained_bool
        ans_vals = m.loc[ans_scorable, "evidence_recall_v2"].dropna()
        ans_prec = m.loc[ans_scorable, "evidence_precision_v2"].dropna()
        ans_len = m.loc[ans_scorable, "predicted_span_len_frac_of_note"].dropna()
        report["--- answered_rows_only (excludes 0-recall abstentions) ---"] = "---"
        report["n_scorable_and_answered"] = int(ans_scorable.sum())
        report["mean_recall__answered_only"] = round(float(ans_vals.mean()), 4) if len(ans_vals) else None
        report["mean_precision__answered_only"] = round(float(ans_prec.mean()), 4) if len(ans_prec) else None
        report["mean_span_len_frac__answered_only"] = round(float(ans_len.mean()), 4) if len(ans_len) else None
    print(json.dumps(report, indent=2))
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
