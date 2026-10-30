#!/usr/bin/env python3
"""How much of the appended omission notice survives BERT's 128-token truncation?

For each notice-appended test control: tokens of the clean note, tokens with the
notice, and how many notice tokens fit in the 126 content positions. When none
fit, the control's model input is identical to the clean input, so the gate's
decision must equal its clean decision.

  python check_notice_truncation.py --controls output/synthetic_dataset/evidence_removed_only.csv \
    --predictions .../provenance_first_availability_supervised_unavailable_test_predictions.csv \
    --clean_predictions .../provenance_first_availability_supervised_clean_test_predictions.csv
"""
import argparse
import pandas as pd
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--controls", required=True)
ap.add_argument("--predictions", required=True)
ap.add_argument("--clean_predictions", default=None)
ap.add_argument("--max_len", type=int, default=128)
ap.add_argument("--model_name", default="emilyalsentzer/Bio_ClinicalBERT")
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model_name)
yes = lambda s: s.astype(str).str.lower().isin(["true", "1"])

c = pd.read_csv(a.controls)
c = c[(c.variant == "mask_temporal_cue") & ~yes(c.masked_cue)]
p = pd.read_csv(a.predictions)
m = p.merge(c[["note_id", "base_note_id", "text"]].rename(columns={"note_id": "example_id"}), on="example_id")
notes = pd.read_csv(a.controls.replace("evidence_removed_only.csv", "synthetic_notes_v1.csv"))
clean_text = dict(zip(notes.note_id, notes.text))
m["n_clean"] = [len(tok(clean_text[b])["input_ids"]) for b in m.base_note_id]
m["n_notice"] = [len(tok(t)["input_ids"]) for t in m.text]
room = a.max_len - 2
m["notice_tokens_visible"] = [max(0, min(room - (nc - 2), nn - nc)) for nc, nn in zip(m.n_clean, m.n_notice)]
m["notice_total_tokens"] = m.n_notice - m.n_clean
m["answered"] = ~yes(m.abstained)
m["visibility"] = pd.cut(m.notice_tokens_visible / m.notice_total_tokens, [-0.01, 0, 0.999, 1.0],
                         labels=["none", "partial", "full"])
if a.clean_predictions:
    cp = pd.read_csv(a.clean_predictions)
    cp = cp.assign(clean_answered=~yes(cp.abstained))[["example_id", "clean_answered", "answer"]]
    m = m.merge(cp.rename(columns={"example_id": "base_note_id", "answer": "clean_answer"}), on="base_note_id", how="left")
print(pd.crosstab(m.visibility, m.answered, margins=True))
cols = ["example_id", "n_clean", "n_notice", "notice_tokens_visible", "notice_total_tokens", "answered", "answer"]
cols += ["clean_answer"] if a.clean_predictions else []
print(m.loc[m.visibility != "full", cols].to_string(index=False))
