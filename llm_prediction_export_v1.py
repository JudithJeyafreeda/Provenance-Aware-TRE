#!/usr/bin/env python3
"""Run an OpenAI-compatible clinical LLM and export prediction-level CSVs.

The script reads synthetic_notes_v1.csv, creates patient-level train/dev/test
splits, queries both an ungated baseline and an evidence-gated configuration,
and writes prediction-level CSVs compatible with evaluate_three_claims.py.


"""

import argparse
import json
import os
import random
import re
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from openai import OpenAI


RELATIONS = {
    "EVENT1-BEFORE-EVENT2",
    "EVENT2-BEFORE-EVENT1",
    "EVENT1-OVERLAP-EVENT2",
}

SYSTEM_PROMPT = """You extract a temporal relation from a clinical note.
Return JSON only, with exactly these keys:
{
  \"relation\": \"EVENT1-BEFORE-EVENT2\" | \"EVENT2-BEFORE-EVENT1\" | \"EVENT1-OVERLAP-EVENT2\" | null,
  \"evidence_span\": string | null,
  \"confidence\": number,
  \"reason\": string
}
The evidence_span must be a contiguous verbatim substring of the supplied
note. Do not paraphrase it. If the relation or supporting text is unclear,
return null for relation and evidence_span.
"""


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)


def parse_json(content: str) -> Dict[str, Any]:
    text = content.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S)
    try:
        obj = json.loads(text)
    except Exception:
        return {"relation": None, "evidence_span": None, "confidence": 0.0,
                "reason": "invalid_json"}
    if not isinstance(obj, dict):
        return {"relation": None, "evidence_span": None, "confidence": 0.0,
                "reason": "non_object_json"}
    relation = obj.get("relation")
    if relation not in RELATIONS:
        relation = None
    span = obj.get("evidence_span")
    if not isinstance(span, str) or not span.strip():
        span = None
    try:
        confidence = max(0.0, min(1.0, float(obj.get("confidence", 0.0))))
    except Exception:
        confidence = 0.0
    return {"relation": relation, "evidence_span": span, "confidence": confidence,
            "reason": str(obj.get("reason", ""))}


def call(client, model, note, temperature):
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=300,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": note},
        ],
    )
    return parse_json(response.choices[0].message.content)


def evaluate_response(note, result, min_chars, min_frac):
    span = result.get("evidence_span")
    valid = isinstance(span, str) and span in note
    coverage = len(span) / max(1, len(note)) if valid else 0.0
    enough = valid and len(span) >= min_chars and coverage >= min_frac
    result = dict(result)
    result.update({"span_valid": bool(valid), "coverage": coverage,
                   "coverage_pass": bool(enough)})
    return result


def necessity_probe(client, model, note, base_result, n_probes, agreement_threshold, temperature):
    if not base_result.get("relation"):
        return {"agreement": 0.0, "pass": False}
    variants = [
        note + "\nFocus only on the cited evidence and relation.",
        "Evidence-focused review:\n" + note,
        note + "\nDo not infer beyond explicit temporal language.",
    ][:n_probes]
    matches = 0
    for variant in variants:
        r = call(client, model, variant, temperature)
        if r.get("relation") == base_result.get("relation"):
            matches += 1
    agreement = matches / max(1, len(variants))
    return {"agreement": agreement, "pass": agreement >= agreement_threshold}


def make_row(row, model_name, result, gated, gate_reason=None):
    relation = result.get("relation")
    gold = row["gold_relation"]
    abstained = relation is None or (gated and gate_reason is not None)
    answer = None if abstained else relation
    correct = None if abstained else int(answer == gold)
    return {
        "example_id": row["note_id"],
        "model": model_name,
        "answer": answer,
        "abstained": abstained,
        "correct": correct,
        "evidence_score": float(result.get("coverage", 0.0)),
        "evidence_sufficient": None,
        "contradiction_present": None,
        "provenance_recorded": True,
        "gate_reason": gate_reason or result.get("reason"),
        "coverage": float(result.get("coverage", 0.0)),
        "evidence_recall": None,
        "evidence_span_valid": bool(result.get("span_valid", False)),
    }


def split_notes(notes, seed):
    notes = notes[notes["gold_relation"].notna()].copy()
    notes = notes[notes["gold_evidence_span"].notna()].copy()
    pids = list(notes["patient_id"].unique())
    random.Random(seed).shuffle(pids)
    n_test = max(1, int(0.15 * len(pids)))
    n_dev = max(1, int(0.15 * len(pids)))
    test = notes[notes.patient_id.isin(pids[:n_test])].copy()
    dev = notes[notes.patient_id.isin(pids[n_test:n_test + n_dev])].copy()
    train = notes[notes.patient_id.isin(pids[n_test + n_dev:])].copy()
    return train, dev, test


def run_split(client, model, df, out_dir, split, args):
    baseline_rows, gated_rows = [], []
    diagnostics = []
    for _, row in df.iterrows():
        note = str(row["text"])
        try:
            base_raw = call(client, model, note, args.temperature)
        except Exception as exc:
            base_raw = {"relation": None, "evidence_span": None, "confidence": 0.0,
                        "reason": f"api_error:{type(exc).__name__}"}
        base = evaluate_response(note, base_raw, args.min_coverage_chars, args.min_coverage_frac)
        baseline_rows.append(make_row(row, "llm_baseline", base, False))

        gate_reason = None
        if base.get("relation") is None:
            gate_reason = "no_relation"
        elif not base.get("coverage_pass"):
            gate_reason = "insufficient_evidence_span"
        else:
            try:
                nec = necessity_probe(
                    client, model, note, base, args.necessity_probes,
                    args.necessity_agreement_threshold, args.temperature
                )
            except Exception:
                nec = {"agreement": 0.0, "pass": False}
            if not nec["pass"]:
                gate_reason = "necessity_check_failed"
            base["necessity_agreement"] = nec["agreement"]
        gated_rows.append(make_row(row, "llm_provenance_first", base, True, gate_reason))
        diagnostics.append({
            "example_id": row["note_id"],
            "split": split,
            "baseline": base,
            "gate_reason": gate_reason,
        })

    pd.DataFrame(baseline_rows).to_csv(os.path.join(out_dir, f"llm_baseline_{split}_predictions.csv"), index=False)
    pd.DataFrame(gated_rows).to_csv(os.path.join(out_dir, f"llm_provenance_first_{split}_predictions.csv"), index=False)
    with open(os.path.join(out_dir, f"llm_{split}_diagnostics.json"), "w") as f:
        json.dump(diagnostics, f, indent=2, default=str)
    print(f"{split}: wrote {len(baseline_rows)} baseline and {len(gated_rows)} gated rows")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", default="llm_predictions")
    p.add_argument("--api_base", default="http://localhost:8000/v1")
    p.add_argument("--api_key", default=os.getenv("VLLM_API_KEY", "EMPTY"))
    p.add_argument("--model", default="mistralai/Mistral-Small-3.2-24B-Instruct-2506")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--min_coverage_chars", type=int, default=10)
    p.add_argument("--min_coverage_frac", type=float, default=0.03)
    p.add_argument("--necessity_probes", type=int, default=3)
    p.add_argument("--necessity_agreement_threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    seed_all(args.seed)
    client = OpenAI(base_url=args.api_base, api_key=args.api_key)
    notes = pd.read_csv(os.path.join(args.data_dir, "synthetic_notes_v1.csv"))
    train, dev, test = split_notes(notes, args.seed)
    run_split(client, args.model, dev, args.out_dir, "dev", args)
    run_split(client, args.model, test, args.out_dir, "test", args)
    with open(os.path.join(args.out_dir, "llm_run_metadata.json"), "w") as f:
        json.dump({"config": vars(args), "train_n": len(train), "dev_n": len(dev), "test_n": len(test)}, f, indent=2)


if __name__ == "__main__":
    main()
