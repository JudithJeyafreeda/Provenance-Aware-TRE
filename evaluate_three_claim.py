#!/usr/bin/env python3



from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


REQUIRED_PREDICTION_COLUMNS = {
    "example_id", "model", "answer", "abstained", "correct", "evidence_score"
}


def nan_to_string(obj: Any) -> Any:
    """Recursively convert NaN, infinity, None, and pandas.NA to 'NaN'."""
    if isinstance(obj, dict):
        return {key: nan_to_string(value) for key, value in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [nan_to_string(value) for value in obj]

    if obj is None or obj is pd.NA:
        return "NaN"

    if isinstance(obj, (float, np.floating)):
        if not np.isfinite(obj):
            return "NaN"

    if isinstance(obj, (int, np.integer, bool, str)):
        return obj

    try:
        if pd.isna(obj):
            return "NaN"
    except (TypeError, ValueError):
        pass

    return obj


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) != 0
    values = series.astype(str).str.strip().str.lower()
    return values.isin({"1", "true", "yes", "y", "abstain", "abstained"})


def validate_prediction_df(df: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = REQUIRED_PREDICTION_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")

    out = df.copy()
    out["abstained"] = as_bool(out["abstained"])
    out["evidence_score"] = pd.to_numeric(out["evidence_score"], errors="coerce")
    out["correct"] = pd.to_numeric(out["correct"], errors="coerce")

    if out["evidence_score"].isna().any():
        raise ValueError(f"{name}: evidence_score contains missing/non-numeric values")

    out["answered"] = ~out["abstained"]
    out["correct_answered"] = np.where(
        out["answered"],
        out["correct"].fillna(0).astype(float) == 1,
        False,
    )
    out["model"] = out["model"].astype(str)
    return out


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> List[float]:
    if n == 0:
        return [np.nan, np.nan]

    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def risk_coverage_curve(
    df: pd.DataFrame,
    score_col: str = "evidence_score",
    grid: int = 101,
) -> pd.DataFrame:
    thresholds = np.unique(
        np.concatenate(([0.0], np.linspace(0.0, 1.0, grid)))
    )
    rows = []

    for threshold in thresholds:
        selected = df["answered"] & (df[score_col] >= threshold)
        n_answered = int(selected.sum())
        n_total = len(df)
        n_correct = int(df.loc[selected, "correct_answered"].sum()) if n_answered else 0
        coverage = n_answered / n_total if n_total else np.nan
        accuracy = n_correct / n_answered if n_answered else np.nan
        risk = 1 - accuracy if n_answered else np.nan

        rows.append({
            "threshold": float(threshold),
            "n_answered": n_answered,
            "coverage": coverage,
            "selective_accuracy": accuracy,
            "selective_risk": risk,
        })

    return pd.DataFrame(rows).sort_values("coverage").reset_index(drop=True)


def aurc(curve: pd.DataFrame) -> float:
    valid = curve.dropna(
        subset=["coverage", "selective_risk"]
    ).sort_values("coverage")

    if len(valid) < 2:
        return np.nan

    if hasattr(np, "trapezoid"):
        area = np.trapezoid(
            valid["selective_risk"].to_numpy(),
            valid["coverage"].to_numpy(),
        )
    else:
        area = np.trapz(
            valid["selective_risk"].to_numpy(),
            valid["coverage"].to_numpy(),
        )

    return float(area)


def choose_threshold(
    dev_df: pd.DataFrame,
    target_coverage: float,
) -> Tuple[float, str, Dict[str, float]]:
    curve = risk_coverage_curve(dev_df)
    valid = curve.dropna(
        subset=["coverage", "selective_risk"]
    ).sort_values("coverage")

    if valid.empty:
        raise ValueError("No non-empty answered set exists on development data.")

    maximum_coverage = float(valid["coverage"].max())

    if maximum_coverage < target_coverage:
        row = valid.iloc[-1]
        return (
            float(row["threshold"]),
            "target_unattainable_maximum_coverage_used",
            {
                "requested_coverage": float(target_coverage),
                "maximum_available_coverage": maximum_coverage,
            },
        )

    eligible = valid[valid["coverage"] >= target_coverage]
    row = eligible.sort_values(
        ["selective_risk", "threshold"]
    ).iloc[0]

    return (
        float(row["threshold"]),
        "dev_min_risk_at_target_coverage",
        {
            "requested_coverage": float(target_coverage),
            "maximum_available_coverage": maximum_coverage,
        },
    )


def apply_threshold_metrics(df: pd.DataFrame, threshold: float) -> Dict[str, Any]:
    selected = df["answered"] & (df["evidence_score"] >= threshold)
    n_total = len(df)
    n_answered = int(selected.sum())
    n_correct = int(df.loc[selected, "correct_answered"].sum()) if n_answered else 0
    coverage = n_answered / n_total if n_total else np.nan
    accuracy = n_correct / n_answered if n_answered else np.nan
    risk = 1 - accuracy if n_answered else np.nan

    result = {
        "threshold": float(threshold),
        "n_total": int(n_total),
        "n_answered": n_answered,
        "coverage": coverage,
        "selective_accuracy": accuracy,
        "selective_risk": risk,
        "accuracy_95_ci_wilson": wilson_interval(n_correct, n_answered),
        "existing_abstention_rate": float(df["abstained"].mean()),
    }

    if "evidence_sufficient" in df.columns:
        evidence = pd.to_numeric(
            df["evidence_sufficient"],
            errors="coerce",
        )
        labeled = evidence.notna()

        if labeled.any():
            sufficient = evidence.eq(1) & labeled
            insufficient = evidence.eq(0) & labeled
            result["evidence_labels"] = {
                "n_labeled": int(labeled.sum()),
                "n_sufficient": int(sufficient.sum()),
                "n_insufficient": int(insufficient.sum()),
                "sufficient_answer_rate": (
                    float(selected[sufficient].mean())
                    if sufficient.any() else np.nan
                ),
                "insufficient_abstention_rate": (
                    float((~selected[insufficient]).mean())
                    if insufficient.any() else np.nan
                ),
                "insufficient_unsafe_answer_rate": (
                    float(
                        ((selected & insufficient) & ~df["correct_answered"]).sum()
                        / insufficient.sum()
                    )
                    if insufficient.any() else np.nan
                ),
            }
        else:
            result["evidence_labels"] = {
                "status": "not_evaluable",
                "reason": "No non-missing evidence_sufficient labels.",
                "n_labeled": 0,
            }

    return result


def random_matched_coverage(
    df: pd.DataFrame,
    target_coverage: float,
    repeats: int,
    seed: int = 7,
) -> Dict[str, Any]:
    n_total = len(df)
    if n_total == 0:
        return {"status": "not_evaluable", "reason": "Empty dataset."}

    n_select = max(1, int(round(target_coverage * n_total)))
    n_select = min(n_select, n_total)
    rng = np.random.default_rng(seed)
    accuracies = []
    answered_counts = []

    for _ in range(repeats):
        selected_idx = rng.choice(n_total, size=n_select, replace=False)
        selected = df.iloc[selected_idx]
        answered = selected["answered"]
        answered_counts.append(int(answered.sum()))
        accuracies.append(
            float(selected.loc[answered, "correct_answered"].mean())
            if answered.any() else 0.0
        )

    return {
        "target_coverage": float(target_coverage),
        "n_selected": int(n_select),
        "random_full_dataset_mean_accuracy_on_answered_selected": float(np.mean(accuracies)),
        "random_full_dataset_accuracy_95_ci": [
            float(np.percentile(accuracies, 2.5)),
            float(np.percentile(accuracies, 97.5)),
        ],
        "random_full_dataset_mean_answered_cases": float(np.mean(answered_counts)),
        "n_repeats": int(repeats),
    }


def mechanism_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    n = len(df)
    reasons_present = (
        df["gate_reason"].fillna("").astype(str).str.len() > 0
        if "gate_reason" in df.columns
        else pd.Series(False, index=df.index)
    )
    provenance_recorded = (
        pd.to_numeric(df["provenance_recorded"], errors="coerce")
        .fillna(0).astype(bool)
        if "provenance_recorded" in df.columns
        else pd.Series(False, index=df.index)
    )

    by_model = {}
    for model, group in df.groupby("model"):
        by_model[str(model)] = {
            "n": int(len(group)),
            "explicit_abstention_field_rate": 1.0,
            "abstention_rate": float(group["abstained"].mean()),
            "gate_reason_rate": float(reasons_present.loc[group.index].mean()),
            "provenance_recorded_rate": float(provenance_recorded.loc[group.index].mean()),
            "answer_has_evidence_score_rate": float(
                group.loc[group["answered"], "evidence_score"].notna().mean()
            ) if group["answered"].any() else np.nan,
        }

    return {
        "n": int(n),
        "explicit_abstention_field_rate": 1.0,
        "gate_reason_rate": float(reasons_present.mean()),
        "provenance_recorded_rate": float(provenance_recorded.mean()),
        "by_model": by_model,
        "interpretation": (
            "These metrics establish that the output format records gate and "
            "provenance states. They do not establish selective-prediction utility."
        ),
    }


def selective_metrics(
    df: pd.DataFrame,
    target_coverage: float,
    threshold: Optional[float],
    threshold_source: str,
    coverage_diagnostic: Dict[str, float],
    random_repeats: int,
) -> Dict[str, Any]:
    curve = risk_coverage_curve(df)
    valid_curve = curve.dropna(
        subset=["coverage", "selective_risk"]
    )

    by_model = {}
    for model, group in df.groupby("model"):
        group_curve = risk_coverage_curve(group)
        by_model[str(model)] = {
            "n": int(len(group)),
            "aurc": aurc(group_curve),
            "at_selected_threshold": (
                apply_threshold_metrics(group, threshold)
                if threshold is not None else None
            ),
        }

    return {
        "risk_coverage_curve": valid_curve.to_dict(orient="records"),
        "aurc": aurc(curve),
        "selected_threshold": (
            apply_threshold_metrics(df, threshold)
            if threshold is not None else None
        ),
        "random_matched_coverage_control": random_matched_coverage(
            df, target_coverage, random_repeats
        ),
        "threshold_source": threshold_source,
        "coverage_selection": coverage_diagnostic,
        "threshold_selection_warning": threshold_source != "dev_min_risk_at_target_coverage",
        "by_model": by_model,
    }


def clinical_usefulness(workflow_path: Optional[str]) -> Dict[str, Any]:
    if not workflow_path:
        return {
            "status": "not_evaluated",
            "reason": "No independent workflow CSV supplied.",
            "required_outcomes": [
                "blinded reviewer or clinician task accuracy",
                "unsafe-error rate",
                "time per case",
                "escalation or review burden",
                "inter-rater agreement",
                "external or prospective validation",
            ],
        }

    df = pd.read_csv(workflow_path)
    required = {"system", "case_id", "task_success", "unsafe_error", "time_seconds"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Workflow CSV missing required columns: {sorted(missing)}")

    for column in ["task_success", "unsafe_error", "time_seconds"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    output = {"status": "evaluated", "by_system": {}}
    for system, group in df.groupby("system"):
        success = group["task_success"].dropna().astype(float).to_numpy()
        unsafe = group["unsafe_error"].dropna().astype(float).to_numpy()
        times = group["time_seconds"].dropna().astype(float).to_numpy()
        output["by_system"][str(system)] = {
            "n_cases": int(len(group)),
            "task_success_rate": float(np.mean(success)) if len(success) else np.nan,
            "task_success_95_ci_wilson": (
                wilson_interval(int(success.sum()), len(success))
                if len(success) else [np.nan, np.nan]
            ),
            "unsafe_error_rate": float(np.mean(unsafe)) if len(unsafe) else np.nan,
            "unsafe_error_95_ci_wilson": (
                wilson_interval(int(unsafe.sum()), len(unsafe))
                if len(unsafe) else [np.nan, np.nan]
            ),
            "mean_time_seconds": float(np.mean(times)) if len(times) else np.nan,
            "median_time_seconds": float(np.median(times)) if len(times) else np.nan,
        }
    return output


def load_predictions(path: str, name: str) -> pd.DataFrame:
    return validate_prediction_df(pd.read_csv(path), name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev_predictions", default=None)
    parser.add_argument("--test_predictions", required=True)
    parser.add_argument("--workflow_csv", default=None)
    parser.add_argument("--out_dir", default="evaluation_results")
    parser.add_argument("--target_coverage", type=float, default=0.50)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--random_repeats", type=int, default=2000)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    test = load_predictions(args.test_predictions, "test_predictions")
    dev = (
        load_predictions(args.dev_predictions, "dev_predictions")
        if args.dev_predictions else None
    )

    if args.threshold is not None:
        threshold = args.threshold
        threshold_source = "user_fixed_threshold"
        coverage_diagnostic = {
            "requested_coverage": float(args.target_coverage),
            "maximum_available_coverage": np.nan,
        }
    elif dev is not None:
        threshold, threshold_source, coverage_diagnostic = choose_threshold(
            dev,
            args.target_coverage,
        )
    else:
        threshold = None
        threshold_source = "descriptive_no_dev_threshold"
        coverage_diagnostic = {
            "requested_coverage": float(args.target_coverage),
            "maximum_available_coverage": np.nan,
        }

    results = {
        "metadata": {
            "target_coverage": args.target_coverage,
            "selected_threshold": threshold,
            "threshold_source": threshold_source,
            "test_file": args.test_predictions,
            "dev_file": args.dev_predictions,
        },
        "mechanism_validation": mechanism_metrics(test),
        "selective_prediction_validation": selective_metrics(
            test,
            args.target_coverage,
            threshold,
            threshold_source,
            coverage_diagnostic,
            args.random_repeats,
        ),
        "clinical_usefulness": clinical_usefulness(args.workflow_csv),
    }

    # Convert all NaN/None values to the string "NaN" before serializing.
    json_results = nan_to_string(results)
    json_path = os.path.join(args.out_dir, "three_claims_evaluation.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2, allow_nan=False)

    summary_rows = []
    for model, group in test.groupby("model"):
        selected = (
            apply_threshold_metrics(group, threshold)
            if threshold is not None else {
                "coverage": float(group["answered"].mean()),
                "selective_accuracy": (
                    float(group.loc[group["answered"], "correct_answered"].mean())
                    if group["answered"].any() else np.nan
                ),
                "selective_risk": np.nan,
            }
        )
        summary_rows.append({
            "model": model,
            "n": len(group),
            "existing_abstention_rate": float(group["abstained"].mean()),
            "coverage_at_selected_rule": selected["coverage"],
            "selective_accuracy_at_selected_rule": selected["selective_accuracy"],
            "selective_risk_at_selected_rule": selected["selective_risk"],
            "mean_evidence_score": float(group["evidence_score"].mean()),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        os.path.join(args.out_dir, "selective_summary.csv"),
        index=False,
    )

    mechanism_rows = []
    for model, values in results["mechanism_validation"]["by_model"].items():
        mechanism_rows.append({"model": model, **values})
    pd.DataFrame(mechanism_rows).to_csv(
        os.path.join(args.out_dir, "mechanism_summary.csv"),
        index=False,
    )

    print(json.dumps({
        "selected_threshold": threshold,
        "threshold_source": threshold_source,
        "coverage_selection": coverage_diagnostic,
        "mechanism_validation": json_results["mechanism_validation"],
        "clinical_usefulness_status": json_results["clinical_usefulness"]["status"],
    }, indent=2, allow_nan=False))
    print(f"Results written to {json_path}")


if __name__ == "__main__":
    main()
