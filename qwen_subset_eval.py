#!/usr/bin/env python3
"""Score both gates on the note sets validated by the Qwen judge. No model calls.

Reads the ID lists written by `judge_qwen.py report` (base note IDs) and the
existing prediction files, and reports, for each set and gate, the same paired
quantities as Table 2: n, clean coverage and accuracy on the same notes, and
coverage and accuracy on the edited notes, with Wilson 95% intervals.
BERT is restricted to the seed-7 test patients automatically (its prediction
files contain only those).

  python qwen_subset_eval.py --notes output/synthetic_dataset/synthetic_notes_v1.csv \
    --idlists qwen_validation \
    --llm_dir output/synthetic_dataset/llm_v2 \
    --bert_dir output/synthetic_dataset/availability_supervised_bert_v3_export \
    --out_csv qwen_subset_results.csv
"""
import argparse
import json
import math
import os

import pandas as pd

SETS = {  # id list -> which edited file it applies to
    "removal_strict_qwen": "removal",
    "removal_strict_both_judges": "removal",
    "removal_implied_qwen": "removal",
    "removal_implied_both_judges": "removal",
    "removal_all_qwen": "removal",
    "keep_valid_qwen": "keep",
}
G = "provenance_first_availability_supervised"


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def load(path, gold, suffix=""):
    df = pd.read_csv(path)
    df["example_id"] = df["example_id"].astype(str)
    df["base"] = df["example_id"].str.replace(suffix + "$", "", regex=True) if suffix else df["example_id"]
    df["abst"] = df["abstained"].astype(str).str.lower().isin(["true", "1", "1.0"])
    df["gold"] = df["base"].map(gold)
    df["ok"] = (~df["abst"]) & (df["answer"].astype(str) == df["gold"])
    return df


def summ(df):
    n = len(df)
    a = int((~df.abst).sum())
    c = int(df.ok.sum())
    return n, a, c


def fmt(k, n):
    if n == 0:
        return "", float("nan")
    lo, hi = wilson(k, n)
    return f"{k / n:.3f} [{lo:.3f}, {hi:.3f}]", k / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", required=True)
    ap.add_argument("--idlists", required=True)
    ap.add_argument("--llm_dir", required=True)
    ap.add_argument("--bert_dir", required=True)
    ap.add_argument("--out_csv", default="qwen_subset_results.csv")
    a = ap.parse_args()

    notes = pd.read_csv(a.notes)
    notes = notes[notes.gold_relation.notna()]
    gold = dict(zip(notes.note_id.astype(str), notes.gold_relation))

    files = {
        "LLM": {"clean": os.path.join(a.llm_dir, "llm_availability_gated_synthetic_notes_relation_only_predictions.csv"),
                "removal": os.path.join(a.llm_dir, "llm_availability_gated_silent_removal_controls_predictions.csv"),
                "keep": os.path.join(a.llm_dir, "llm_availability_gated_silent_keep_controls_predictions.csv")},
        "BERT": {"clean": os.path.join(a.bert_dir, f"{G}_clean_test_predictions.csv"),
                 "removal": os.path.join(a.bert_dir, f"{G}_silent_removal_controls_test_predictions.csv"),
                 "keep": os.path.join(a.bert_dir, f"{G}_silent_keep_controls_test_predictions.csv")},
    }
    rows = []
    for gate, fp in files.items():
        clean = load(fp["clean"], gold)
        edited = {"removal": load(fp["removal"], gold, "_silent_removal"),
                  "keep": load(fp["keep"], gold, "_silent_keep")}
        for set_name, kind in SETS.items():
            path = os.path.join(a.idlists, set_name + ".json")
            if not os.path.exists(path):
                continue
            ids = set(map(str, json.load(open(path))))
            e = edited[kind][edited[kind].base.isin(ids)]
            c = clean[clean.example_id.isin(set(e.base))]
            n, ans, cor = summ(e)
            cn, cans, ccor = summ(c)
            cov_s, _ = fmt(ans, n)
            acc_s, _ = fmt(cor, ans)
            rows.append({
                "gate": gate, "set": set_name, "ids_in_list": len(ids), "n": n,
                "clean_coverage": round(cans / cn, 3) if cn else None,
                "clean_accuracy": round(ccor / cans, 3) if cans else None,
                "edited_coverage": cov_s, "edited_accuracy": acc_s,
                "edited_answered": ans, "edited_correct": cor,
            })
    out = pd.DataFrame(rows)
    out.to_csv(a.out_csv, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(out.to_string(index=False))
    print(f"\nwrote {a.out_csv}")


if __name__ == "__main__":
    main()
