#!/usr/bin/env python3
"""Run an explicitly availability-gated LLM on clean/control CSVs.

"""

import argparse
import json
import os
import re
from typing import Any, Dict

import numpy as np
import pandas as pd
from openai import OpenAI

RELATIONS = {
    "EVENT1-BEFORE-EVENT2",
    "EVENT2-BEFORE-EVENT1",
    "EVENT1-OVERLAP-EVENT2",
}

SYSTEM = """Return JSON only:
{
  \"relation\": \"EVENT1-BEFORE-EVENT2\" | \"EVENT2-BEFORE-EVENT1\" | \"EVENT1-OVERLAP-EVENT2\" | null,
  \"evidence_span\": string | null,
  \"confidence\": number,
  \"reason\": string
}
The evidence_span must be copied verbatim from the note. If the note does not
contain clear temporal evidence, return null for relation and evidence_span.
"""


def parse(content: str) -> Dict[str, Any]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.I | re.S)
    try:
        obj = json.loads(text)
    except Exception:
        return {"relation": None, "evidence_span": None, "confidence": 0.0, "reason": "invalid_json"}
    return {
        "relation": obj.get("relation") if obj.get("relation") in RELATIONS else None,
        "evidence_span": obj.get("evidence_span") if isinstance(obj.get("evidence_span"), str) else None,
        "confidence": float(obj.get("confidence", 0.0)),
        "reason": str(obj.get("reason", "")),
    }


def call(client, model, text, temperature):
    r = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=300,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": text},
        ],
    )
    return parse(r.choices[0].message.content)


def necessity(client, model, text, relation, probes, threshold, temperature):
    if relation is None:
        return 0.0
    prompts = [
        text + "\nFocus only on explicit temporal evidence.",
        "Evidence-focused review:\n" + text,
        text + "\nDo not infer beyond the note.",
    ][:probes]
    matches = 0
    for prompt in prompts:
        out = call(client, model, prompt, temperature)
        matches += int(out.get("relation") == relation)
    return matches / max(1, len(prompts))


def run(df, client, args, model_name):
    rows = []
    for _, row in df.iterrows():
        text = str(row["text"])
        try:
            out = call(client, args.model, text, args.temperature)
        except Exception as exc:
            out = {"relation": None, "evidence_span": None, "confidence": 0.0, "reason": f"api_error:{type(exc).__name__}"}

        span = out.get("evidence_span")
        valid = isinstance(span, str) and span in text
        coverage = len(span) / max(1, len(text)) if valid else 0.0
        reason = None
        if out.get("relation") is None:
            reason = "no_relation"
        elif not valid or len(span) < args.min_coverage_chars or coverage < args.min_coverage_frac:
            reason = "insufficient_evidence_span"
        else:
            agreement = necessity(
                client, args.model, text, out["relation"], args.necessity_probes,
                args.necessity_agreement_threshold, args.temperature,
            )
            if agreement < args.necessity_agreement_threshold:
                reason = "necessity_check_failed"
            out["necessity_agreement"] = agreement

        abstained = reason is not None
        answer = None if abstained else out.get("relation")
        gold = row["gold_relation"]
        rows.append({
            "example_id": str(row["note_id"]),
            "model": model_name,
            "answer": answer,
            "abstained": abstained,
            "correct": None if abstained else int(answer == gold),
            "evidence_score": coverage,
            "availability_probability": float(row.get("evidence_available", np.nan)),
            "evidence_sufficient": row.get("evidence_available", np.nan),
            "contradiction_present": row.get("contradiction_present_oracle", np.nan),
            "provenance_recorded": True,
            "gate_reason": reason,
            "coverage": coverage,
            "evidence_case": row.get("evidence_case", None),
            "evidence_span_valid": valid,
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", required=True)
    p.add_argument("--out_dir", default="llm_availability_predictions")
    p.add_argument("--api_base", default="http://localhost:8000/v1")
    p.add_argument("--api_key", default=os.getenv("VLLM_API_KEY", "EMPTY"))
    p.add_argument("--model", default="mistralai/Mistral-Small-3.2-24B-Instruct-2506")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--min_coverage_chars", type=int, default=10)
    p.add_argument("--min_coverage_frac", type=float, default=0.03)
    p.add_argument("--necessity_probes", type=int, default=3)
    p.add_argument("--necessity_agreement_threshold", type=float, default=0.5)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.input_csv)
    required = {"note_id", "text", "gold_relation"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {sorted(missing)}")
    client = OpenAI(base_url=args.api_base, api_key=args.api_key)
    stem = os.path.splitext(os.path.basename(args.input_csv))[0]
    result = run(df, client, args, "llm_provenance_first_availability_gated")
    path = os.path.join(args.out_dir, f"llm_availability_gated_{stem}_predictions.csv")
    result.to_csv(path, index=False)
    with open(os.path.join(args.out_dir, f"llm_availability_gated_{stem}_metadata.json"), "w") as f:
        json.dump({"config": vars(args), "n": len(df)}, f, indent=2)
    print(f"wrote {len(result)} rows to {path}")


if __name__ == "__main__":
    main()
