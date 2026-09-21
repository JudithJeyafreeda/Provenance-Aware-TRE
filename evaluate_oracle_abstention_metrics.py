#!/usr/bin/env python3
"""Evaluate abstention against generator-derived synthetic oracle labels.

Inputs
------
1. A prediction CSV produced by bert_prediction_export_v1.py or
   llm_prediction_export_v1.py.
2. synthetic_notes_v1.csv from synthetic_longitudinal_generator_v1.py.
3. Optionally synthetic_adversarial_v1.csv.

Prediction CSV required columns:
  example_id, model, answer, abstained, correct, evidence_score

Generator-derived labels
-------------------------
For each base note:
  evidence_sufficient_oracle = 1 when a non-empty gold evidence span exists
  in the evaluated note and the note has a gold relation.

For adversarial notes:
  contradiction_present_oracle = 1 when injected_spans or
  contradiction_types is non-empty.
  FOr BERT and LLM seperately run:
python evaluate_oracle_abstention_metrics.py \
  --predictions bert_provenance_first_test_predictions.csv \
  --notes synthetic_notes_v1.csv \
  --adversarial synthetic_adversarial_v1.csv \
  --out_dir oracle_abstention_bert
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


REQUIRED_PREDICTION_COLUMNS = {
    "example_id", "model", "answer", "abstained", "correct", "evidence_score"
}


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {
        "1", "true", "yes", "y", "abstain", "abstained"
    }


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> List[float]:
    if n == 0:
        return [np.nan, np.nan]
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def json_safe(value: Any) -> Any:
    """Convert non-finite values to the JSON string 'NaN'."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if value is None or value is pd.NA:
        return "NaN"
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return "NaN"
    return value


def has_injected_content(value: Any) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    text = str(value).strip()
    if text in {"", "[]", "nan", "None"}:
        return False
    try:
        parsed = json.loads(text)
        return isinstance(parsed, list) and len(parsed) > 0
    except Exception:
        return True


def validate_predictions(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_PREDICTION_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Prediction CSV missing columns: {sorted(missing)}")

    out = df.copy()
    out["example_id"] = out["example_id"].astype(str)
    out["abstained_bool"] = out["abstained"].map(parse_bool)
    out["answered"] = ~out["abstained_bool"]
    out["correct_num"] = pd.to_numeric(out["correct"], errors="coerce")
    out["correct_answered"] = np.where(
        out["answered"], out["correct_num"].fillna(0).eq(1), False
    )
    return out


def clean_oracle_labels(notes_path: str) -> pd.DataFrame:
    notes = pd.read_csv(notes_path)
    notes["example_id"] = notes["note_id"].astype(str)

    relation_present = notes["gold_relation"].notna()
    span_present = (
        notes["gold_evidence_span"].notna()
        & notes["gold_evidence_span"].astype(str).str.strip().ne("")
    )

    notes["evidence_sufficient_oracle"] = (
        relation_present & span_present
    ).astype(int)

    def exact_alignment(row):
        if not bool(row["evidence_sufficient_oracle"]):
            return False
        return (
            isinstance(row["text"], str)
            and str(row["gold_evidence_span"]) in str(row["text"])
        )

    notes["gold_span_located_exactly"] = notes.apply(exact_alignment, axis=1)
    notes["evidence_alignment_status"] = np.where(
        notes["evidence_sufficient_oracle"].eq(0),
        "no_generator_evidence_label",
        np.where(
            notes["gold_span_located_exactly"],
            "exact",
            "not_exactly_located",
        ),
    )
    notes["contradiction_present_oracle"] = 0
    notes["oracle_source"] = "synthetic_notes_v1_generator"

    return notes[[
        "example_id", "evidence_sufficient_oracle",
        "contradiction_present_oracle", "evidence_alignment_status",
        "oracle_source",
    ]].drop_duplicates("example_id", keep="last")


def adversarial_oracle_labels(adversarial_path: str) -> pd.DataFrame:
    adv = pd.read_csv(adversarial_path)
    adv["example_id"] = adv["note_id"].astype(str)
    adv["contradiction_present_oracle"] = adv.apply(
        lambda row: int(
            has_injected_content(row.get("injected_spans"))
            or has_injected_content(row.get("contradiction_types"))
        ),
        axis=1,
    )
    adv["evidence_sufficient_oracle"] = np.nan
    adv["evidence_alignment_status"] = "adversarial_not_clean_evidence_label"
    adv["oracle_source"] = "synthetic_adversarial_v1_generator"
    return adv[[
        "example_id", "evidence_sufficient_oracle",
        "contradiction_present_oracle", "evidence_alignment_status",
        "oracle_source",
    ]].drop_duplicates("example_id", keep="last")


def load_oracle_labels(notes_path: str, adversarial_path: Optional[str]) -> pd.DataFrame:
    clean = clean_oracle_labels(notes_path)
    if adversarial_path and os.path.exists(adversarial_path):
        adversarial = adversarial_oracle_labels(adversarial_path)
        return pd.concat([clean, adversarial], ignore_index=True).drop_duplicates(
            "example_id", keep="last"
        )
    return clean


def compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    n = len(df)
    abstain = df["abstained_bool"]
    answered = df["answered"]

    evidence_known = df["evidence_sufficient_oracle"].notna()
    sufficient = evidence_known & df["evidence_sufficient_oracle"].eq(1)
    insufficient = evidence_known & df["evidence_sufficient_oracle"].eq(0)
    contradiction = df["contradiction_present_oracle"].eq(1)

    output: Dict[str, Any] = {
        "n": int(n),
        "abstention_rate": float(abstain.mean()) if n else np.nan,
        "coverage": float(answered.mean()) if n else np.nan,
        "oracle_label_counts": {
            "evidence_sufficient": int(sufficient.sum()),
            "evidence_insufficient": int(insufficient.sum()),
            "evidence_unlabeled": int((~evidence_known).sum()),
            "contradiction_present": int(contradiction.sum()),
        },
        "alignment_status_counts": (
            df["evidence_alignment_status"].value_counts(dropna=False).to_dict()
            if "evidence_alignment_status" in df.columns else {}
        ),
        "oracle_label_status": "generator_derived_not_human_annotated",
    }

    if sufficient.any():
        answer_rate = answered[sufficient].mean()
        output["sufficient_evidence_answer_rate"] = float(answer_rate)
        output["sufficient_evidence_false_abstention_rate"] = float(1 - answer_rate)
    else:
        output["sufficient_evidence_answer_rate"] = np.nan
        output["sufficient_evidence_false_abstention_rate"] = np.nan

    if insufficient.any():
        output["insufficient_evidence_abstention_recall"] = float(abstain[insufficient].mean())
        output["insufficient_evidence_unsafe_answer_rate"] = float(
            (answered[insufficient] & ~df.loc[insufficient, "correct_answered"]).mean()
        )
    else:
        output["insufficient_evidence_abstention_recall"] = np.nan
        output["insufficient_evidence_unsafe_answer_rate"] = np.nan

    if contradiction.any():
        output["contradiction_abstention_rate"] = float(abstain[contradiction].mean())
        output["contradiction_unsafe_answer_rate"] = float(
            (answered[contradiction] & ~df.loc[contradiction, "correct_answered"]).mean()
        )
    else:
        output["contradiction_abstention_rate"] = np.nan
        output["contradiction_unsafe_answer_rate"] = np.nan

    if answered.any():
        correct = int(df.loc[answered, "correct_answered"].sum())
        n_answered = int(answered.sum())
        output["answered_accuracy"] = float(correct / n_answered)
        output["answered_accuracy_95_ci_wilson"] = wilson(correct, n_answered)
    else:
        output["answered_accuracy"] = np.nan
        output["answered_accuracy_95_ci_wilson"] = [np.nan, np.nan]

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--notes", required=True)
    parser.add_argument("--adversarial", default=None)
    parser.add_argument("--out_dir", default="oracle_abstention_results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    predictions = validate_predictions(pd.read_csv(args.predictions))
    labels = load_oracle_labels(args.notes, args.adversarial)
    merged = predictions.merge(labels, on="example_id", how="left")

    overall = compute_metrics(merged)
    by_model = {
        str(model): compute_metrics(group)
        for model, group in merged.groupby("model")
    }

    result = {
        "metadata": {
            "predictions": args.predictions,
            "notes": args.notes,
            "adversarial": args.adversarial,
            "oracle_type": "generator_derived",
            "human_annotation": False,
        },
        "overall": overall,
        "by_model": by_model,
        "interpretation": (
            "These metrics evaluate abstention against generator-derived labels. "
            "They support synthetic mechanism validation only and do not establish "
            "clinical evidence sufficiency or clinical usefulness."
        ),
    }

    merged.to_csv(
        os.path.join(args.out_dir, "oracle_labeled_predictions.csv"),
        index=False,
    )
    with open(os.path.join(args.out_dir, "oracle_abstention_metrics.json"), "w") as f:
        json.dump(json_safe(result), f, indent=2, allow_nan=False)

    print(json.dumps(json_safe(result), indent=2, allow_nan=False))
    print(f"Wrote {args.out_dir}/oracle_abstention_metrics.json")
    print(f"Wrote {args.out_dir}/oracle_labeled_predictions.csv")


if __name__ == "__main__":
    main()
