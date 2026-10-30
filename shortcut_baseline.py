#!/usr/bin/env python3
"""No-learning shortcut baseline for the clean task (paper Section 3.1).

Rule: if the note mentions exactly one day number -> EVENT1-OVERLAP-EVENT2;
otherwise, if the treatment is mentioned before the diagnosis -> EVENT2-BEFORE-EVENT1;
otherwise EVENT1-BEFORE-EVENT2. Concept mentions are located with the fact values
from synthetic_facts_v1.csv (falling back to their first 4+ letter word).

This is a reconstruction. With it the committed corpus gives 0.667 overall,
0.667 on the seed-7 test split and 0.996 on structured-admission notes; the paper
reports 0.651 / 0.633 / 0.996. Replace locate() / predict() with the rule that
produced the paper's numbers and commit that version.

  python shortcut_baseline.py --data_dir output/synthetic_dataset
"""
import argparse, json, os, random, re
import pandas as pd

E12, E21, OVL = "EVENT1-BEFORE-EVENT2", "EVENT2-BEFORE-EVENT1", "EVENT1-OVERLAP-EVENT2"


def locate(text, phrase):
    t, p = text.lower(), str(phrase).lower()
    i = t.find(p)
    if i >= 0:
        return i
    for w in re.findall(r"[a-z]{4,}", p):
        i = t.find(w)
        if i >= 0:
            return i
    return None


def first(text, values):
    pos = [p for p in (locate(text, v) for v in values) if p is not None]
    return min(pos) if pos else None


def predict(text, dx, tx):
    n_day = len(re.findall(r"\bday\s*(\d+)", text, flags=re.I))
    if n_day == 1:
        return OVL
    pd_, pt = first(text, dx), first(text, tx)
    if pt is not None and (pd_ is None or pt < pd_):
        return E21
    return E12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    notes = pd.read_csv(os.path.join(a.data_dir, "synthetic_notes_v1.csv"))
    facts = pd.read_csv(os.path.join(a.data_dir, "synthetic_facts_v1.csv")).set_index("fact_id")
    rel = notes[notes.gold_relation.notna() & notes.gold_evidence_span.notna()].reset_index(drop=True)
    preds = []
    for _, r in rel.iterrows():
        ids = [i for i in json.loads(r.mentioned_fact_ids) if i in facts.index]
        f = facts.loc[ids]
        preds.append(predict(r.text, f[f.concept_type == "diagnosis"].value, f[f.concept_type == "treatment"].value))
    rel["pred"] = preds
    rel["ok"] = rel.pred == rel.gold_relation
    pats = list(rel.patient_id.unique())
    random.Random(a.seed).shuffle(pats)
    test = set(pats[: int(0.15 * len(pats))])
    print(f"overall accuracy      {rel.ok.mean():.3f}  (paper 0.651)")
    print(f"test-split accuracy   {rel[rel.patient_id.isin(test)].ok.mean():.3f}  (paper 0.633)")
    print("by style:", rel.groupby('style').ok.mean().round(3).to_dict(), "(paper structured_admission 0.996)")
    print("majority class        %.3f" % (rel.gold_relation.value_counts().max() / len(rel)))


if __name__ == "__main__":
    main()
