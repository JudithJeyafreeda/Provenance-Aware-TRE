#!/usr/bin/env python3
"""Evaluate abstention on clean and evidence-unavailable control predictions.

Inputs:
  --clean_predictions: prediction CSV on original evidence-available notes
  --control_predictions: prediction CSV on evidence-removed/masked notes
  --control_csv: evidence_removed_only.csv created by
                 make_evidence_unavailable_adversarial.py


"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

import numpy as np
import pandas as pd


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {
        "1", "true", "yes", "y", "abstain", "abstained"
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if value is None or value is pd.NA:
        return "NaN"
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return "NaN"
    return value


def prepare_predictions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"example_id", "model", "abstained", "correct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df = df.copy()
    df["example_id"] = df["example_id"].astype(str)
    df["abstained_bool"] = df["abstained"].map(parse_bool)
    df["answered"] = ~df["abstained_bool"]
    df["correct_num"] = pd.to_numeric(df["correct"], errors="coerce")
    df["correct_answered"] = np.where(
        df["answered"], df["correct_num"].fillna(0).eq(1), False
    )
    return df


def prepare_control_labels(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"note_id", "evidence_available_oracle", "evidence_case"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    labels = df.copy()
    labels["example_id"] = labels["note_id"].astype(str)
    keep = ["example_id", "evidence_available_oracle", "evidence_case"]
    if "contradiction_present_oracle" in labels.columns:
        keep.append("contradiction_present_oracle")
    return labels[keep].drop_duplicates("example_id")


def summarize_clean(df: pd.DataFrame) -> Dict[str, Any]:
    n = len(df)
    answered = df["answered"]
    abstain = df["abstained_bool"]
    return {
        "n": int(n),
        "coverage": float(answered.mean()) if n else np.nan,
        "false_abstention_rate": float(abstain.mean()) if n else np.nan,
        "answered_accuracy": float(df.loc[answered, "correct_answered"].mean()) if answered.any() else np.nan,
        "n_answered": int(answered.sum()),
        "n_abstained": int(abstain.sum()),
    }


def summarize_control(df: pd.DataFrame) -> Dict[str, Any]:
    n = len(df)
    answered = df["answered"]
    abstain = df["abstained_bool"]
    return {
        "n": int(n),
        "abstention_recall": float(abstain.mean()) if n else np.nan,
        "coverage_on_unavailable_evidence": float(answered.mean()) if n else np.nan,
        "unsafe_answer_rate": float(
            (answered & ~df["correct_answered"]).mean()
        ) if n else np.nan,
        "answered_accuracy": float(df.loc[answered, "correct_answered"].mean()) if answered.any() else np.nan,
        "n_answered": int(answered.sum()),
        "n_abstained": int(abstain.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_predictions", required=True)
    parser.add_argument("--control_predictions", required=True)
    parser.add_argument("--control_csv", required=True)
    parser.add_argument("--out_dir", default="control_abstention_results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    clean = prepare_predictions(args.clean_predictions)
    control = prepare_predictions(args.control_predictions)
    labels = prepare_control_labels(args.control_csv)
    if "evidence_case" not in control.columns:
        control["evidence_case"] = control["example_id"].str.extract(
        r"_(remove_evidence|mask_temporal_cue)$",
        expand=False,
    )

        control["evidence_case"] = control["evidence_case"].fillna(
        "evidence_unavailable"
    )

    if control.empty:
        raise ValueError(
            "No control prediction IDs matched the control CSV. "
            "Check that prediction example_id values preserve the suffixes "
            "_remove_evidence and _mask_temporal_cue."
        )

    result = {
        "metadata": {
            "clean_predictions": args.clean_predictions,
            "control_predictions": args.control_predictions,
            "control_csv": args.control_csv,
            "oracle_type": "generator_controlled",
            "human_annotation": False,
        },
        "by_model": {},
        "interpretation": (
            "Correct abstention means high clean coverage with low false abstention "
            "and high abstention recall with low unsafe-answer rate on evidence-unavailable "
            "controls. These are synthetic generator-controlled results, not clinical validation."
        ),
    }

    models = sorted(set(clean["model"].unique()) | set(control["model"].unique()))
    rows = []
    for model in models:
        clean_m = clean[clean["model"] == model]
        control_m = control[control["model"] == model]
        model_result = {
            "clean": summarize_clean(clean_m),
            "control_overall": summarize_control(control_m),
            "control_by_variant": {
                str(variant): summarize_control(group)
                for variant, group in control_m.groupby("evidence_case")
            },
        }
        result["by_model"][str(model)] = model_result
        rows.append({
            "model": model,
            "clean_coverage": model_result["clean"]["coverage"],
            "clean_false_abstention_rate": model_result["clean"]["false_abstention_rate"],
            "control_abstention_recall": model_result["control_overall"]["abstention_recall"],
            "control_unsafe_answer_rate": model_result["control_overall"]["unsafe_answer_rate"],
            "control_coverage": model_result["control_overall"]["coverage_on_unavailable_evidence"],
        })

    result = json_safe(result)
    with open(os.path.join(args.out_dir, "control_abstention_metrics.json"), "w") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    pd.DataFrame(rows).to_csv(
        os.path.join(args.out_dir, "control_abstention_summary.csv"),
        index=False,
    )
    control.to_csv(
        os.path.join(args.out_dir, "control_predictions_with_labels.csv"),
        index=False,
    )

    print(json.dumps(result, indent=2, allow_nan=False))
    print(f"Wrote {args.out_dir}/control_abstention_metrics.json")
    print(f"Wrote {args.out_dir}/control_abstention_summary.csv")


if __name__ == "__main__":
    main()
