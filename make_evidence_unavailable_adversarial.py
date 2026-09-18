#!/usr/bin/env python3
"""Create evidence-unavailable adversarial controls for abstention testing.

This script reads synthetic_notes_v1.csv and creates matched variants where
the relation-bearing evidence is removed, masked, or made contradictory.
It writes rows with explicit generator-controlled labels:

  evidence_available_oracle = 0
  contradiction_present_oracle = 0 or 1
  evidence_case

Variants:
  remove_evidence       Replace the gold evidence sentence with a neutral one.
  mask_temporal_cue     Remove temporal cue words and replace them with [MASK].
  contradiction         Append a contradiction that reverses or questions the
                        temporal relation.
  remove_and_contradict Remove evidence and append a contradiction.
python make_evidence_unavailable_adversarial.py \
  --notes /mnt/eds_projets/eds_iam/Judith_ARIPPA/work/LLM/TEMPORALEX/output/synthetic_dataset/synthetic_notes_v1.csv \
  --out_csv /mnt/eds_projets/eds_iam/Judith_ARIPPA/work/LLM/TEMPORALEX/output/synthetic_dataset/evidence_removed_only.csv \
  --variants remove_evidence,mask_temporal_cue
Contradiction Testing: python make_evidence_unavailable_adversarial.py \
  --notes /mnt/eds_projets/eds_iam/Judith_ARIPPA/work/LLM/TEMPORALEX/output/synthetic_dataset/synthetic_notes_v1.csv \
  --out_csv /mnt/eds_projets/eds_iam/Judith_ARIPPA/work/LLM/TEMPORALEX/output/synthetic_dataset/contradiction_controls.csv \
  --variants contradiction,remove_and_contradict
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from typing import Any, Dict, List

import pandas as pd


TEMPORAL_CUES = [
    "prior to", "before", "preceding", "following", "after",
    "subsequent to", "during", "concurrent with", "while undergoing",
]


def replace_first(text: str, old: str, new: str) -> tuple[str, bool]:
    if old and old in text:
        return text.replace(old, new, 1), True
    return text, False


def remove_evidence(row: pd.Series) -> tuple[str, Dict[str, Any]]:
    text = str(row["text"])
    span = str(row.get("gold_evidence_span", ""))
    replacement = (
        "The note does not document the temporal relationship between the two events."
    )
    new_text, replaced = replace_first(text, span, replacement)
    if not replaced:
        new_text = (
            "The patient was admitted for evaluation. "
            "The available documentation does not specify the temporal relationship "
            "between the two clinical events."
        )
    return new_text, {
        "evidence_available_oracle": 0,
        "contradiction_present_oracle": 0,
        "evidence_case": "evidence_removed",
        "removed_gold_span": bool(replaced),
    }


def mask_temporal_cue(row: pd.Series) -> tuple[str, Dict[str, Any]]:
    text = str(row["text"])
    masked = text
    count = 0
    for cue in TEMPORAL_CUES:
        pattern = re.compile(re.escape(cue), flags=re.IGNORECASE)
        masked, n = pattern.subn("[MASKED_TEMPORAL_CUE]", masked, count=1)
        count += n
        if count:
            break
    if count == 0:
        masked = (
            masked + " The temporal cue needed to determine the relation was omitted."
        )
    return masked, {
        "evidence_available_oracle": 0,
        "contradiction_present_oracle": 0,
        "evidence_case": "temporal_cue_masked",
        "masked_cue": bool(count),
    }


def contradiction(row: pd.Series) -> tuple[str, Dict[str, Any]]:
    text = str(row["text"])
    clause = (
        " A separate record contradicts the stated ordering; the two events "
        "may have occurred in the opposite temporal order."
    )
    return text + clause, {
        "evidence_available_oracle": 0,
        "contradiction_present_oracle": 1,
        "evidence_case": "contradiction_injected",
        "injected_clause": clause.strip(),
    }


def remove_and_contradict(row: pd.Series) -> tuple[str, Dict[str, Any]]:
    text, _ = remove_evidence(row)
    clause = (
        " A separate record contradicts the proposed ordering, so the temporal "
        "relation cannot be determined reliably."
    )
    return text + clause, {
        "evidence_available_oracle": 0,
        "contradiction_present_oracle": 1,
        "evidence_case": "evidence_removed_and_contradiction",
        "injected_clause": clause.strip(),
    }


VARIANTS = {
    "remove_evidence": remove_evidence,
    "mask_temporal_cue": mask_temporal_cue,
    "contradiction": contradiction,
    "remove_and_contradict": remove_and_contradict,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--notes", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--variants", default="remove_evidence,mask_temporal_cue,contradiction,remove_and_contradict")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    notes = pd.read_csv(args.notes)
    notes = notes[notes["gold_relation"].notna()].copy()
    notes = notes[notes["gold_evidence_span"].notna()].copy()

    variant_names = [x.strip() for x in args.variants.split(",") if x.strip()]
    unknown = set(variant_names) - set(VARIANTS)
    if unknown:
        raise ValueError(f"Unknown variants: {sorted(unknown)}")

    if not 0 < args.fraction <= 1:
        raise ValueError("--fraction must be in (0, 1].")

    if args.fraction < 1:
        n = max(1, int(round(len(notes) * args.fraction)))
        notes = notes.sample(n=n, random_state=args.seed)

    rows: List[Dict[str, Any]] = []
    for _, row in notes.iterrows():
        for variant_name in variant_names:
            text, labels = VARIANTS[variant_name](row)
            record = row.to_dict()
            record.update(labels)
            record["example_id"] = f"{row['note_id']}_{variant_name}"
            record["note_id"] = record["example_id"]
            record["base_note_id"] = row["note_id"]
            record["text"] = text
            record["variant"] = variant_name
            record["gold_evidence_span"] = ""
            record["injected_spans"] = json.dumps([])
            record["contradiction_types"] = json.dumps(
                [variant_name] if labels["contradiction_present_oracle"] else []
            )
            rows.append(record)

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    diagnostics = {
        "source_notes": args.notes,
        "n_source_notes": int(len(notes)),
        "n_output_variants": int(len(out)),
        "variants": variant_names,
        "evidence_available_oracle": 0,
        "contradiction_variant_counts": out.groupby("variant")["contradiction_present_oracle"].first().to_dict(),
    }
    diag_path = os.path.splitext(args.out_csv)[0] + "_diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)

    print(json.dumps(diagnostics, indent=2))
    print(f"Wrote adversarial controls to {args.out_csv}")
    print(f"Wrote diagnostics to {diag_path}")


if __name__ == "__main__":
    main()
