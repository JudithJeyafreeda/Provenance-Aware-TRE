"""
Provenance-First Clinical NLP: Proof-of-Concept with Bio_ClinicalBERT
========================================================================

Requirements (run outside the hosted sandbox -- needs internet/GPU access):
    pip install torch transformers pandas numpy scipy networkx
Dependency: provenance_temporal_kg.py must be in the same directory 

Usage
-----
    CUDA_VISIBLE_DEVICES="" python provenance_first_poc_clinicalBERT.py --data_dir "path/to/data" --epochs 20 --run_certification

"""

import re
import os
import json
import random
import argparse
import pickle
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

from provenance_temporal_kg import (
    ProvenanceTemporalKG, ingest_bert_result, aggregate_temporal_motifs,
)

# ----------------------------------------------------------------------------
# 1. Synthetic dataset generation (unchanged from v5 -- order-balanced,
#    per-template diagnostics, position-control set all still active)
# ----------------------------------------------------------------------------

CONDITIONS = [
    "type 2 diabetes mellitus", "acute heart failure", "community-acquired pneumonia",
    "chronic kidney disease stage 3", "atrial fibrillation", "hypertensive crisis",
    "acute appendicitis", "sepsis secondary to UTI", "COPD exacerbation", "deep vein thrombosis",
    "acute pancreatitis", "pulmonary embolism", "cellulitis of the lower extremity",
    "gastrointestinal bleed", "acute ischemic stroke", "hyperosmolar hyperglycemic state",
    "acute cholecystitis", "bacterial meningitis", "myocardial infarction", "acute kidney injury",
]
TREATMENTS = [
    "intravenous furosemide", "empiric ceftriaxone", "insulin sliding scale",
    "anticoagulation with apixaban", "supplemental oxygen therapy", "laparoscopic appendectomy",
    "nebulized bronchodilators", "antihypertensive titration", "fluid resuscitation", "beta-blocker therapy",
    "broad-spectrum antibiotics", "thrombolytic therapy", "packed red blood cell transfusion",
    "insulin infusion protocol", "cholecystectomy", "lumbar puncture and antibiotics",
    "percutaneous coronary intervention", "hemodialysis", "vasopressor support", "surgical debridement",
]
SYMPTOMS = [
    "shortness of breath", "abdominal pain", "palpitations", "lower extremity edema",
    "productive cough", "fever and chills", "dizziness", "chest tightness", "nausea and vomiting", "fatigue",
    "diaphoresis", "confusion", "hematemesis", "syncope", "jaundice", "back pain",
    "decreased urine output", "headache", "tachycardia", "hypotension",
]
CUES_BEFORE_FORWARD = ["prior to", "before", "preceding", "in advance of"]
CUES_BEFORE_REVERSE = ["which followed", "occurring after", "subsequent to the earlier"]
CUES_AFTER_FORWARD = ["following", "after", "subsequent to", "once"]
CUES_AFTER_REVERSE = ["which preceded", "occurring before", "prior to the later"]
CUES_OVERLAP = ["during", "concurrent with", "while undergoing", "at the same time as"]
DEPARTMENTS = [
    "Cardiology", "Internal Medicine", "Emergency Department", "Nephrology", "Pulmonology",
    "Gastroenterology", "Neurology", "General Surgery", "Infectious Disease", "Critical Care",
]
RELATIONS = ["BEFORE", "AFTER", "OVERLAP"]


def make_note(note_id, relation_type, adversarial=False, natural_variant=False, order="random"):
    condition = random.choice(CONDITIONS)
    treatment = random.choice(TREATMENTS)
    symptom = random.choice(SYMPTOMS)
    dept = random.choice(DEPARTMENTS)
    day1 = random.randint(1, 14)
    day2 = day1 + random.randint(1, 6)
    event1 = f"the patient was diagnosed with {condition}"
    event2 = f"{treatment} was initiated"

    chosen_order = random.choice(["forward", "reverse"]) if order == "random" else order

    if relation_type == "BEFORE":
        if chosen_order == "forward":
            cue = random.choice(CUES_BEFORE_FORWARD)
            sentence = (f"On day {day1}, {event1}. {cue.capitalize()} initiating {treatment} "
                        f"on day {day2}, the patient reported {symptom}.")
        else:
            cue = random.choice(CUES_BEFORE_REVERSE)
            sentence = (f"{treatment.capitalize()} was initiated on day {day2}, {cue} "
                        f"day {day1} diagnosis of {condition}, with associated {symptom}.")
        gold_relation = "EVENT1-BEFORE-EVENT2"

    elif relation_type == "AFTER":
        if chosen_order == "forward":
            cue = random.choice(CUES_AFTER_FORWARD)
            sentence = (f"{event2.capitalize()} on day {day1}. {cue.capitalize()} treatment, "
                        f"{event1} on day {day2}, correlating with persistent {symptom}.")
        else:
            cue = random.choice(CUES_AFTER_REVERSE)
            sentence = (f"On day {day2}, {event1}, {cue} day {day1} initiation of "
                        f"{treatment}, with persistent {symptom}.")
        gold_relation = "EVENT2-BEFORE-EVENT1"

    else:  # OVERLAP
        cue = random.choice(CUES_OVERLAP)
        if chosen_order == "forward":
            sentence = (f"{event1.capitalize()} on day {day1}, and {cue} the diagnostic workup, "
                        f"{event2} to manage associated {symptom}.")
        else:
            sentence = (f"{treatment.capitalize()} was initiated on day {day1}, {cue} the "
                        f"diagnostic evaluation that led to a diagnosis of {condition}, "
                        f"with associated {symptom}.")
        gold_relation = "EVENT1-OVERLAP-EVENT2"

    preamble = f"Admitted to {dept} for evaluation. "
    closing = (f" The patient was monitored on the {dept.lower()} service and remained "
               f"hemodynamically stable throughout the admission.")
    full_text = preamble + sentence + closing

    record = {
        "note_id": note_id, "text": full_text, "gold_relation": gold_relation,
        "gold_evidence_span": sentence, "variant": "base",
        "event1": event1, "event2": event2, "mention_order": chosen_order,
    }

    if adversarial:
        fabricated = (f" Notably, the patient also mentioned {random.choice(SYMPTOMS)} "
                       f"which had resolved spontaneously and was not treated.")
        cut = full_text.rfind(".")
        adv_text = full_text[:cut] + "." + fabricated + full_text[cut + 1:]
        record.update({"text": adv_text, "variant": "adversarial",
                        "fabricated_clause": fabricated.strip()})

    if natural_variant:
        abbrev_text = full_text.replace("intravenous", "IV").replace("day", "hospital day")
        record.update({"text": abbrev_text, "variant": "natural_variability",
                        "gold_evidence_span": sentence.replace("intravenous", "IV").replace("day", "hospital day")})

    return record


def generate_datasets(seed=7, n_base_per_relation=500, n_paired=600, out_dir="output"):
    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    base = [make_note(f"{rel}_{i:04d}_base", rel, order="random")
            for rel in RELATIONS for i in range(n_base_per_relation)]
    paired_idx = random.sample(range(len(base)), n_paired)
    paired = []
    for idx in paired_idx:
        src = base[idx]
        rel_key = src["note_id"].split("_")[0]
        paired.append(make_note(src["note_id"].replace("_base", "_adv"), rel_key,
                                 adversarial=True, order="random"))
        paired.append(make_note(src["note_id"].replace("_base", "_nat"), rel_key,
                                 natural_variant=True, order="random"))
    random.shuffle(base)
    n = len(base)
    train_r = base[:int(0.7 * n)]
    dev_r = base[int(0.7 * n):int(0.85 * n)]
    test_r = base[int(0.85 * n):]

    def to_df(records):
        return pd.DataFrame([{
            "note_id": r["note_id"], "text": r["text"], "gold_relation": r["gold_relation"],
            "gold_evidence_span": r["gold_evidence_span"], "variant": r["variant"],
            "event1": r["event1"], "event2": r["event2"],
            "mention_order": r.get("mention_order", "forward"),
        } for r in records])

    df_train, df_dev, df_test = to_df(train_r), to_df(dev_r), to_df(test_r)
    df_manip = to_df(paired)
    df_train.to_csv(os.path.join(out_dir, "synthetic_clinical_train.csv"), index=False)
    df_dev.to_csv(os.path.join(out_dir, "synthetic_clinical_dev.csv"), index=False)
    df_test.to_csv(os.path.join(out_dir, "synthetic_clinical_test.csv"), index=False)
    df_manip.to_csv(os.path.join(out_dir, "synthetic_clinical_manipulation_pairs.csv"), index=False)
    return df_train, df_dev, df_test, df_manip


def generate_position_control_set(df_test, n_per_relation=30, seed=99):
    random.seed(seed)
    swapped_rows = []
    for rel in RELATIONS:
        subset = df_test[df_test["note_id"].str.startswith(rel)]
        n = min(n_per_relation, len(subset))
        subset = subset.sample(n=n, random_state=seed)
        for _, row in subset.iterrows():
            original_order = row.get("mention_order", "forward")
            forced_order = "reverse" if original_order == "forward" else "forward"
            new_rec = make_note(row["note_id"] + "_posctrl", rel, order=forced_order)
            swapped_rows.append(new_rec)
    return pd.DataFrame([{
        "note_id": r["note_id"], "text": r["text"], "gold_relation": r["gold_relation"],
        "gold_evidence_span": r["gold_evidence_span"], "variant": r["variant"],
        "event1": r["event1"], "event2": r["event2"], "mention_order": r["mention_order"],
    } for r in swapped_rows])


def load_external_dataset(data_dir, test_frac=0.15, dev_frac=0.15, seed=7):
    """Loads notes/adversarial CSVs produced by
    synthetic_longitudinal_generator_v1.py and reshapes them into the
    train/dev/test/manip dataframes ClinicalRelationDataset and
    run_manipulation_eval() expect, as an alternative to this script's own
    generate_datasets().

    Only rows with a non-null gold_relation are usable for the 3-way
    relation classifier -- in the v1 generator that is exactly the
    admission notes (revision and follow-up notes don't carry a
    gold_relation/gold_evidence_span pair and are written out for future
    RO2 graph-level use instead, not for this classifier).

    Split is by patient_id rather than note_id: a patient's several notes
    should not be split across train/dev/test once graph-level tasks are
    added later, even though only one note per patient is used here today.

    generate_position_control_set() is NOT called for external data: the
    v1 generator currently only renders admission sentences in forward
    event order (it has no CUES_*_REVERSE analogue), so there is no
    order-swapped variant to construct yet. Interpret per_template_abstain
    diagnostics from evaluate() with that gap in mind until the generator
    is extended to emit a reverse-order variant.
    """
    notes_path = os.path.join(data_dir, "synthetic_notes_v1.csv")
    adv_path = os.path.join(data_dir, "synthetic_adversarial_v1.csv")
    if not os.path.exists(notes_path):
        raise FileNotFoundError(
            f"{notes_path} not found. Run synthetic_longitudinal_generator_v1.py "
            f"--out_dir {data_dir} first.")

    df_all_notes = pd.read_csv(notes_path)
    n_notes_total = len(df_all_notes)

    df_rel = df_all_notes[df_all_notes["gold_relation"].notna()].copy()
    if "mention_order" not in df_rel.columns:
        df_rel["mention_order"] = "forward"  # generator v1 only emits forward-order admission sentences
    dropped = df_rel["gold_evidence_span"].isna().sum()
    if dropped:
        print(f"  [warn] {dropped} admission notes have no valid gold_evidence_span "
              f"(likely an LLM-paraphrase fallback in the generator) and will be "
              f"excluded from the relation-classification split.")
    df_rel = df_rel[df_rel["gold_evidence_span"].notna()].reset_index(drop=True)

    patient_ids = sorted(df_rel["patient_id"].unique())
    rng = random.Random(seed)
    rng.shuffle(patient_ids)
    n = len(patient_ids)
    n_test = max(1, int(test_frac * n))
    n_dev = max(1, int(dev_frac * n))
    test_pids = set(patient_ids[:n_test])
    dev_pids = set(patient_ids[n_test:n_test + n_dev])
    train_pids = set(patient_ids[n_test + n_dev:])

    df_train = df_rel[df_rel["patient_id"].isin(train_pids)].reset_index(drop=True)
    df_dev = df_rel[df_rel["patient_id"].isin(dev_pids)].reset_index(drop=True)
    df_test = df_rel[df_rel["patient_id"].isin(test_pids)].reset_index(drop=True)

    df_manip = pd.DataFrame()
    if os.path.exists(adv_path):
        df_adv = pd.read_csv(adv_path)
        df_adv = df_adv[df_adv["base_note_id"].isin(df_rel["note_id"])].reset_index(drop=True)
        if "mention_order" not in df_adv.columns:
            df_adv["mention_order"] = "forward"
        df_manip = df_adv
    else:
        print(f"  [warn] {adv_path} not found -- manipulation-eval dataframe will be empty "
              f"(run_manipulation_eval will report n=0 for both variants).")

    print(f"  loaded {n_notes_total} total notes ({len(df_rel)} usable for relation "
          f"classification: train={len(df_train)} dev={len(df_dev)} test={len(df_test)}, "
          f"split by patient_id) manip_pairs={len(df_manip)}")
    return df_train, df_dev, df_test, df_manip


def get_fabricated_char_span(text):
    m = re.search(r"Notably,.*?not treated\.", text)
    return (m.start(), m.end()) if m else None


def resolve_base_note_id(row):
    """Returns the note_id of the un-perturbed base note this adversarial
    row derives from. Handles both this script's own generate_datasets()
    naming convention (note_id encodes _base/_adv/_nat suffixes) and
    synthetic_longitudinal_generator_v1.py's explicit base_note_id column,
    which is authoritative when present."""
    explicit = row.get("base_note_id")
    if isinstance(explicit, str) and explicit:
        return explicit
    return row["note_id"].replace("_adv", "_base").replace("_nat", "_base")


def get_injected_char_spans(row):
    """Returns a list of (start, end) char spans for injected/adversarial
    content in this row's text. Prefers the explicit injected_spans JSON
    column written by synthetic_longitudinal_generator_v1.py, which
    supports multiple, typed contradiction spans (fabricated_symptom,
    temporal_contradiction, dosage_contradiction, negation_flip). Falls
    back to the single hardcoded fabricated-clause regex used by this
    script's own generate_datasets() for backward compatibility -- without
    this fallback-aware dispatch, evidence_shift would silently read as 0
    for every non-fabricated_symptom contradiction type, since the old
    regex only matches that one clause shape."""
    raw = row.get("injected_spans")
    if isinstance(raw, str) and raw.strip() and raw.strip() != "[]":
        try:
            spans = json.loads(raw)
            return [(s["start"], s["end"]) for s in spans]
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    span = get_fabricated_char_span(row["text"])
    return [span] if span else []


def get_injected_token_mask(text, offsets, tokenizer, char_spans):
    mask = [0] * len(offsets)
    for (fs, fe) in char_spans:
        for i, (s, e) in enumerate(offsets):
            if s == e:
                continue
            if s >= fs and e <= fe:
                mask[i] = 1
    return mask


# ----------------------------------------------------------------------------
# 2. Tokenization / evidence alignment (unchanged from v5)
# ----------------------------------------------------------------------------

def build_evidence_labels_bert(text, gold_span, tokenizer, max_len=128):
    enc = tokenizer(text, truncation=True, max_length=max_len, return_offsets_mapping=True,
                     return_tensors=None)
    offsets = enc["offset_mapping"]
    span_start = text.find(gold_span)
    span_end = span_start + len(gold_span) if span_start != -1 else -1

    labels = []
    for (s, e) in offsets:
        if s == e:
            labels.append(0)
        elif span_start != -1 and s >= span_start and e <= span_end:
            labels.append(1)
        else:
            labels.append(0)
    return enc["input_ids"], enc["attention_mask"], labels, offsets


def get_fabricated_token_mask(text, offsets, tokenizer):
    span = get_fabricated_char_span(text)
    mask = [0] * len(offsets)
    if span is None:
        return mask
    fs, fe = span
    for i, (s, e) in enumerate(offsets):
        if s == e:
            continue
        if s >= fs and e <= fe:
            mask[i] = 1
    return mask


class ClinicalRelationDataset(Dataset):
    def __init__(self, df, tokenizer, label2idx, max_len=128):
        self.rows = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.label2idx = label2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        input_ids, attn_mask, ev_labels, offsets = build_evidence_labels_bert(
            row["text"], row["gold_evidence_span"], self.tokenizer, self.max_len)
        return {
            "input_ids": input_ids, "attention_mask": attn_mask, "ev_labels": ev_labels,
            "label_idx": self.label2idx[row["gold_relation"]], "note_id": row["note_id"],
            "text": row["text"], "gold_evidence_span": row["gold_evidence_span"],
            "event1": row["event1"], "event2": row["event2"],
        }


def collate_fn(batch, pad_token_id=0):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, attn_mask, ev_labels, labels = [], [], [], []
    for b in batch:
        pad = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_token_id] * pad)
        attn_mask.append(b["attention_mask"] + [0] * pad)
        ev_labels.append(b["ev_labels"] + [0] * pad)
        labels.append(b["label_idx"])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
        "ev_labels": torch.tensor(ev_labels, dtype=torch.float),
        "label_idx": torch.tensor(labels, dtype=torch.long),
        "note_ids": [b["note_id"] for b in batch],
        "texts": [b["text"] for b in batch],
        "event1s": [b["event1"] for b in batch],
        "event2s": [b["event2"] for b in batch],
    }


# ----------------------------------------------------------------------------
# 3. Provenance-first model with Bio_ClinicalBERT encoder [AXIS 1]
#    UNCHANGED from v5: gated_forward()'s hard-gate mechanism is exactly as
#    before. The fix in v6 lives entirely in the TRAINING LOSS (Section 4),
#    not in the model or the gate itself.
# ----------------------------------------------------------------------------

class ProvenanceFirstBERT(nn.Module):
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT", n_labels=3,
                 hard_gate=True, min_coverage=0.08, necessity_drop_threshold=None,
                 finetune_bert=False, dropout=0.1, soft_gate_temperature=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        if not finetune_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.evidence_head = nn.Linear(hidden, 1)
        self.pred_head = nn.Linear(hidden, n_labels)
        self.dropout = nn.Dropout(dropout)

        self.hard_gate = hard_gate
        self.min_coverage = min_coverage
        self.necessity_drop_threshold = necessity_drop_threshold
        self.n_labels = n_labels
        self.soft_gate_temperature = soft_gate_temperature

    def encode(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state

    def evidence_scores(self, hidden, attention_mask):
        logits = self.evidence_head(self.dropout(hidden)).squeeze(-1)
        logits = logits.masked_fill(attention_mask == 0, -1e9)
        scores = torch.sigmoid(logits)
        return scores, logits

    def gated_forward(self, input_ids, attention_mask, evidence_threshold=0.5,
                       check_necessity=True, enforce_gate=True):
        hidden = self.encode(input_ids, attention_mask)
        scores, _ = self.evidence_scores(hidden, attention_mask)
        B, T, H = hidden.shape

        results = []
        for b in range(B):
            mask_valid = attention_mask[b].bool()
            s = scores[b][mask_valid]
            h = hidden[b][mask_valid]
            ev_mask = (s >= evidence_threshold).float()
            coverage = ev_mask.mean().item() if ev_mask.numel() > 0 else 0.0
            gate_pass = (coverage >= self.min_coverage) if (self.hard_gate and enforce_gate) else True

            if self.hard_gate and enforce_gate and not gate_pass:
                results.append({"abstain": True, "reason": "low_coverage", "coverage": coverage,
                                 "probs": None, "ev_mask": ev_mask, "ev_scores": s})
                continue

            weights = ev_mask * s if self.hard_gate else s
            wsum = weights.sum()
            w_norm = weights / wsum if wsum > 1e-8 else torch.ones_like(weights) / weights.numel()
            pooled = (w_norm.unsqueeze(-1) * h).sum(dim=0)
            logits = self.pred_head(pooled)
            probs = F.softmax(logits, dim=-1)

            entry = {"abstain": False, "coverage": coverage, "probs": probs,
                      "ev_mask": ev_mask, "ev_scores": s}

            if self.hard_gate and check_necessity and self.necessity_drop_threshold is not None:
                rem_w = 1 - ev_mask
                rem_sum = rem_w.sum()
                w_rem = rem_w / rem_sum if rem_sum > 1e-8 else torch.ones_like(rem_w) / rem_w.numel()
                pooled_rem = (w_rem.unsqueeze(-1) * h).sum(dim=0)
                logits_rem = self.pred_head(pooled_rem)
                probs_rem = F.softmax(logits_rem, dim=-1)
                pred_class = torch.argmax(probs).item()
                drop = (probs[pred_class] - probs_rem[pred_class]).item()
                entry["necessity_drop"] = drop
                if enforce_gate and drop < self.necessity_drop_threshold:
                    entry["abstain"] = True
                    entry["reason"] = "necessity_check_failed"
            results.append(entry)
        return results

    def training_forward_full(self, input_ids, attention_mask, tau=0.5):
        hidden = self.encode(input_ids, attention_mask)
        scores, ev_logits = self.evidence_scores(hidden, attention_mask)
        e_mask_soft = torch.sigmoid((scores - tau) / self.soft_gate_temperature)
        e_mask_soft = e_mask_soft * attention_mask.float()

        B = hidden.shape[0]
        per_example = []
        for b in range(B):
            mask_valid = attention_mask[b].bool()
            h = hidden[b][mask_valid]
            s = scores[b][mask_valid]
            m_soft = e_mask_soft[b][mask_valid]

            w = m_soft
            wsum = w.sum().clamp_min(1e-8)
            h_evidence = (w.unsqueeze(-1) * h).sum(dim=0) / wsum

            with torch.no_grad():
                hard_mask = (s >= tau).float()
                E_idx = torch.nonzero(hard_mask, as_tuple=True)[0]
                nonE_idx = torch.nonzero(1 - hard_mask, as_tuple=True)[0]

            per_example.append({
                "h": h, "s": s, "m_soft": m_soft, "h_evidence": h_evidence,
                "E_idx": E_idx, "nonE_idx": nonE_idx,
            })
        return per_example, ev_logits, attention_mask


# ----------------------------------------------------------------------------
# 4. Training objective: TRAIN_STEP
#    CHANGED IN v6: added the coverage-floor hinge loss (L_floor). This is
#    the core fix for the collapse observed in the v5 run. lambda_evi
#    default lowered 0.002 -> 0.0005 and now gated by --warmup_epochs.
# ----------------------------------------------------------------------------

def leave_one_out_ablation(h, evidence_idx, pred_head, base_logits_target_class):
    if evidence_idx.numel() <= 1:
        return torch.zeros(evidence_idx.numel(), device=h.device)

    effects = []
    full_pooled = h[evidence_idx].mean(dim=0)
    full_probs = F.softmax(pred_head(full_pooled), dim=-1)
    full_p = full_probs[base_logits_target_class]

    for i in range(evidence_idx.numel()):
        keep = torch.cat([evidence_idx[:i], evidence_idx[i + 1:]])
        if keep.numel() == 0:
            effects.append(full_p.detach())
            continue
        pooled_loo = h[keep].mean(dim=0)
        probs_loo = F.softmax(pred_head(pooled_loo), dim=-1)
        p_loo = probs_loo[base_logits_target_class]
        effects.append((full_p - p_loo).detach())
    return torch.stack(effects)


def spearman_corr_torch(x, y):
    if x.numel() < 2:
        return torch.tensor(1.0, device=x.device)
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    if np.allclose(x_np, x_np[0]) or np.allclose(y_np, y_np[0]):
        return torch.tensor(0.0, device=x.device)
    rho, _ = stats.spearmanr(x_np, y_np)
    return torch.tensor(0.0 if np.isnan(rho) else rho, device=x.device)


def train_step(model, input_ids, attention_mask, labels, tau=0.5,
               lambda_evi=0.0005, lambda_faith=0.5, lambda_causal=0.1,
               lambda_coverage_floor=1.0, min_coverage_target=0.10,
               apply_evi_penalty=True, causal_sample_size=6):
    """v6: added lambda_coverage_floor / min_coverage_target hinge term
    (L_floor). apply_evi_penalty=False fully disables L_evi during warmup
    epochs so early training never pressures coverage downward before a
    stable working level is established."""
    per_example, ev_logits, attn_mask = model.training_forward_full(input_ids, attention_mask, tau=tau)
    B = len(per_example)

    pred_losses, evi_losses, suff_losses, nec_losses, causal_losses, floor_losses = [], [], [], [], [], []
    coverages_this_batch = []

    for b in range(B):
        ex = per_example[b]
        h, s, m_soft = ex["h"], ex["s"], ex["m_soft"]
        E_idx, nonE_idx = ex["E_idx"], ex["nonE_idx"]
        coverages_this_batch.append((s >= tau).float().mean().item())

        logits_full = model.pred_head(ex["h_evidence"])
        probs_full = F.softmax(logits_full, dim=-1)
        y_true_b = labels[b]

        pred_losses.append(F.cross_entropy(logits_full.unsqueeze(0), y_true_b.unsqueeze(0)))
        evi_losses.append(m_soft.mean())
        floor_losses.append(F.relu(torch.tensor(min_coverage_target, device=h.device) - m_soft.mean()))

        if E_idx.numel() > 0:
            y_hat_evidence_only = F.softmax(model.pred_head(h[E_idx].mean(dim=0)), dim=-1)
        else:
            y_hat_evidence_only = probs_full
        if nonE_idx.numel() > 0:
            y_hat_non_evidence = F.softmax(model.pred_head(h[nonE_idx].mean(dim=0)), dim=-1)
        else:
            y_hat_non_evidence = probs_full

        l_suff = F.mse_loss(y_hat_evidence_only, probs_full.detach())
        l_nec = -F.mse_loss(y_hat_non_evidence, probs_full.detach())
        suff_losses.append(l_suff)
        nec_losses.append(l_nec)

        if E_idx.numel() >= 2:
            sample_idx = E_idx
            if sample_idx.numel() > causal_sample_size:
                perm = torch.randperm(sample_idx.numel())[:causal_sample_size]
                sample_idx = sample_idx[perm]
            target_class = torch.argmax(probs_full).item()
            loo_effects = leave_one_out_ablation(h, sample_idx, model.pred_head, target_class)
            e_scores_sample = s[sample_idx]
            rho = spearman_corr_torch(e_scores_sample, loo_effects)
            causal_losses.append(1 - rho)
        else:
            causal_losses.append(torch.tensor(0.0, device=h.device))

    L_pred = torch.stack(pred_losses).mean()
    L_evi = (lambda_evi * torch.stack(evi_losses).mean()) if apply_evi_penalty else torch.tensor(0.0)
    L_floor = lambda_coverage_floor * torch.stack(floor_losses).mean()
    L_faith = lambda_faith * (torch.stack(suff_losses).mean() + torch.stack(nec_losses).mean())
    L_causal = lambda_causal * torch.stack(causal_losses).mean()
    L_total = L_pred + L_evi + L_floor + L_faith + L_causal

    return L_total, {
        "L_total": L_total.item(), "L_pred": L_pred.item(),
        "L_evi": L_evi.item() if torch.is_tensor(L_evi) else L_evi,
        "L_floor": L_floor.item(), "L_faith": L_faith.item(), "L_causal": L_causal.item(),
        "mean_train_coverage": float(np.mean(coverages_this_batch)),
    }


def train(model, train_loader, dev_loader, device, epochs=20, lr=5e-5,
          tau=0.5, lambda_evi=0.0005, lambda_faith=0.5, lambda_causal=0.1,
          lambda_coverage_floor=1.0, min_coverage_target=0.10,
          warmup_epochs=6, min_coverage_stop_threshold=None):
    """v6: warmup_epochs now ALSO controls apply_evi_penalty (L_evi is 0.0
    during warmup, not just reduced) and adds automatic early stopping if
    mean_train_coverage falls below min_coverage_stop_threshold after
    warmup ends -- this is the direct fix for the v5 run continuing to
    train for 10 more epochs after the collapse was already locked in."""
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)

    for epoch in range(epochs):
        model.train()
        totals = {"L_total": 0.0, "L_pred": 0.0, "L_evi": 0.0, "L_floor": 0.0,
                   "L_faith": 0.0, "L_causal": 0.0, "mean_train_coverage": 0.0}
        n_batches = 0
        apply_evi_penalty = (epoch >= warmup_epochs)

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["label_idx"].to(device)

            L_total, components = train_step(
                model, input_ids, attn_mask, labels, tau=tau,
                lambda_evi=lambda_evi, lambda_faith=lambda_faith, lambda_causal=lambda_causal,
                lambda_coverage_floor=lambda_coverage_floor, min_coverage_target=min_coverage_target,
                apply_evi_penalty=apply_evi_penalty)
            optimizer.zero_grad()
            L_total.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            for k in totals:
                totals[k] += components[k]
            n_batches += 1

        avg = {k: v / n_batches for k, v in totals.items()}
        enforce_gate_this_epoch = (epoch >= warmup_epochs)
        dev_metrics = evaluate(model, dev_loader, device, enforce_gate=enforce_gate_this_epoch)
        warmup_tag = " [warmup: gate not enforced, L_evi=0]" if not enforce_gate_this_epoch else ""
        print(f"Epoch {epoch+1}/{epochs} | " +
              " ".join(f"{k}={v:.4f}" for k, v in avg.items()) +
              f" | dev_abstain={dev_metrics['abstain_rate']:.3f} "
              f"dev_acc={dev_metrics['accuracy_on_answered']:.3f}{warmup_tag}")
        if dev_metrics.get("shortcut_flag"):
            print(f"  [WARN] dev per-template abstain rates: {dev_metrics['per_template_abstain_rate']}")

        if enforce_gate_this_epoch and min_coverage_stop_threshold is not None:
            if avg["mean_train_coverage"] < min_coverage_stop_threshold:
                print(f"  [warn] mean_train_coverage ({avg['mean_train_coverage']:.4f}) fell below "
                      f"--min_coverage_stop_threshold ({min_coverage_stop_threshold}) at epoch "
                      f"{epoch+1}. Stopping early to avoid further collapse -- this is the same "
                      f"failure mode seen in the v5 run (coverage collapsed to ~0.025-0.028 and "
                      f"stayed there for 10 more epochs). Consider lowering --lambda_evi further "
                      f"(e.g. 0.0002) or raising --lambda_coverage_floor (e.g. 2.0) and rerunning.")
                break
    return model


@torch.no_grad()
def evaluate(model, loader, device, evidence_threshold=0.5, check_necessity=True, enforce_gate=True):
    model.eval()
    correct, total, abstained = 0, 0, 0
    overlaps = []
    coverages = []
    breakdown = {}

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        ev_labels = batch["ev_labels"]
        labels = batch["label_idx"]
        note_ids = batch["note_ids"]

        results = model.gated_forward(input_ids, attn_mask, evidence_threshold, check_necessity,
                                       enforce_gate=enforce_gate)
        for i, r in enumerate(results):
            total += 1
            coverages.append(r["coverage"])
            template = note_ids[i].split("_")[0]
            breakdown.setdefault(template, {"n": 0, "abstained": 0, "correct": 0})
            breakdown[template]["n"] += 1

            if r["abstain"]:
                abstained += 1
                breakdown[template]["abstained"] += 1
                continue
            pred = torch.argmax(r["probs"]).item()
            if pred == labels[i].item():
                correct += 1
                breakdown[template]["correct"] += 1
            gold_ev = ev_labels[i][attn_mask[i].bool().cpu()]
            gold_ev = gold_ev[:r["ev_mask"].numel()]
            if gold_ev.sum().item() > 0:
                ov = (gold_ev.to(device) * r["ev_mask"]).sum().item() / gold_ev.sum().item()
                overlaps.append(ov)

    per_template_abstain_rate = {k: v["abstained"] / v["n"] for k, v in breakdown.items() if v["n"] > 0}
    rates = list(per_template_abstain_rate.values())
    shortcut_flag = (max(rates) - min(rates)) > 0.5 if rates else False
    if shortcut_flag:
        print(f"  [WARN] per-template abstention rates vary by >50 points: "
              f"{per_template_abstain_rate} -- this pattern previously indicated "
              f"the selector keying on sentence template/position rather than content.")

    return {
        "accuracy_on_answered": correct / max(1, total - abstained),
        "abstain_rate": abstained / total,
        "mean_evidence_recall": float(np.mean(overlaps)) if overlaps else 0.0,
        "mean_coverage": float(np.mean(coverages)) if coverages else 0.0,
        "n": total,
        "per_template_breakdown": breakdown,
        "per_template_abstain_rate": per_template_abstain_rate,
        "shortcut_flag": shortcut_flag,
    }


# ----------------------------------------------------------------------------
# 4b. Axis-3 integration: unchanged from v5
# ----------------------------------------------------------------------------

@torch.no_grad()
def build_knowledge_graph(model, loader, device, site_id, evidence_threshold=0.5,
                           check_necessity=True, model_id="bio_clinicalbert"):
    kg = ProvenanceTemporalKG(site_id=site_id)
    model.eval()
    n_ingested = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        note_ids = batch["note_ids"]
        event1s = batch["event1s"]
        event2s = batch["event2s"]

        results = model.gated_forward(input_ids, attn_mask, evidence_threshold, check_necessity,
                                       enforce_gate=True)
        for i, r in enumerate(results):
            ingest_bert_result(
                kg, note_id=note_ids[i], event1_label=event1s[i], event2_label=event2s[i],
                bert_result=r, tau=evidence_threshold, model_id=model_id,
            )
            n_ingested += 1
    print(f"  [kg] ingested {n_ingested} gated_forward() results into site '{site_id}' "
          f"({kg.graph.number_of_nodes()} nodes, {kg.graph.number_of_edges()} edges)")
    return kg


# ----------------------------------------------------------------------------
# 5. Provenance manipulation experiment (unchanged from v5)
# ----------------------------------------------------------------------------

@torch.no_grad()
def run_manipulation_eval(model, df_manip, all_base_lookup, tokenizer, device,
                           evidence_threshold=0.5, check_necessity=True, max_len=128):
    model.eval()
    res = {"adversarial": {"n": 0, "abstain": 0, "pred_stable": 0, "evidence_shift": 0},
           "natural_variability": {"n": 0, "abstain": 0, "pred_stable": 0, "evidence_recall": []}}

    def single_forward(text):
        enc = tokenizer(text, truncation=True, max_length=max_len, return_offsets_mapping=True)
        input_ids = torch.tensor([enc["input_ids"]], dtype=torch.long).to(device)
        attn_mask = torch.tensor([enc["attention_mask"]], dtype=torch.long).to(device)
        results = model.gated_forward(input_ids, attn_mask, evidence_threshold, check_necessity)
        return results[0], enc["offset_mapping"]

    for _, row in df_manip.iterrows():
        out, offsets = single_forward(row["text"])
        base_id = resolve_base_note_id(row)
        base_row = all_base_lookup.get(base_id)
        if base_row is None:
            continue
        base_out, _ = single_forward(base_row["text"])

        if row["variant"] == "adversarial":
            d = res["adversarial"]
            d["n"] += 1
            if out["abstain"]:
                d["abstain"] += 1
                continue
            if not base_out["abstain"] and torch.argmax(out["probs"]).item() == torch.argmax(base_out["probs"]).item():
                d["pred_stable"] += 1
            char_spans = get_injected_char_spans(row)
            fab_mask = get_injected_token_mask(row["text"], offsets, tokenizer, char_spans)
            fab_mask_t = torch.tensor(fab_mask, dtype=torch.float)[:out["ev_mask"].numel()]
            if fab_mask_t.sum().item() > 0 and (fab_mask_t.to(device) * out["ev_mask"]).sum().item() > 0:
                d["evidence_shift"] += 1
        else:
            d = res["natural_variability"]
            d["n"] += 1
            if out["abstain"]:
                d["abstain"] += 1
                continue
            if not base_out["abstain"] and torch.argmax(out["probs"]).item() == torch.argmax(base_out["probs"]).item():
                d["pred_stable"] += 1
            _, _, ev_labels, _ = build_evidence_labels_bert(row["text"], row["gold_evidence_span"], tokenizer, max_len)
            gold_ev = torch.tensor(ev_labels, dtype=torch.float)[:out["ev_mask"].numel()]
            if gold_ev.sum().item() > 0:
                d["evidence_recall"].append(
                    (gold_ev.to(device) * out["ev_mask"]).sum().item() / gold_ev.sum().item())
    return res


def summarize_manipulation(res):
    a, nv = res["adversarial"], res["natural_variability"]
    return {
        "adversarial_n": a["n"], "adversarial_abstain_rate": a["abstain"] / max(1, a["n"]),
        "adversarial_pred_stability": a["pred_stable"] / max(1, a["n"]),
        "adversarial_evidence_shift_rate": a["evidence_shift"] / max(1, a["n"]),
        "natural_n": nv["n"], "natural_abstain_rate": nv["abstain"] / max(1, nv["n"]),
        "natural_pred_stability": nv["pred_stable"] / max(1, nv["n"]),
        "natural_mean_evidence_recall": float(np.mean(nv["evidence_recall"])) if nv["evidence_recall"] else None,
    }


# ----------------------------------------------------------------------------
# 6. Dual robustness / provenance-stability certification: CERTIFY [AXIS 2]
#    CHANGED IN v6: default n_samples raised, ceiling warning added --
#    same Clopper-Pearson-ceiling fix as the LLM POC script.
# ----------------------------------------------------------------------------

ABBREV_MAP = {
    "intravenous": "IV", "day": "d", "diagnosed": "dx'd", "patient": "pt",
    "treatment": "tx", "with": "w/", "and": "&",
}
OCR_CONFUSABLES = {"o": "0", "l": "1", "i": "1", "s": "5", "e": "3", "a": "@"}
SYNONYM_MAP = {
    "initiated": "started", "reported": "noted", "diagnosed": "identified",
    "monitored": "observed", "remained": "stayed", "associated": "related",
}
FILLER_TOKENS = [" also", " subsequently", " notably", " per report"]
PERTURBATION_SPACE = ["abbreviation_swap", "ocr_noise", "token_insertion", "synonym_substitution"]


def _perturb_abbreviation_swap(text, k):
    words = text.split()
    idxs = [i for i, w in enumerate(words) if w.strip(".,").lower() in ABBREV_MAP]
    random.shuffle(idxs)
    for i in idxs[:k]:
        w = words[i]
        core = w.strip(".,").lower()
        suffix = w[len(core):]
        words[i] = ABBREV_MAP.get(core, core) + suffix
    return " ".join(words)


def _perturb_ocr_noise(text, k):
    chars = list(text)
    positions = [i for i, c in enumerate(chars) if c.lower() in OCR_CONFUSABLES]
    random.shuffle(positions)
    for p in positions[:k]:
        c = chars[p]
        repl = OCR_CONFUSABLES[c.lower()]
        chars[p] = repl.upper() if c.isupper() else repl
    return "".join(chars)


def _perturb_token_insertion(text, k):
    words = text.split()
    for _ in range(k):
        if len(words) < 2:
            break
        pos = random.randint(1, len(words) - 1)
        words.insert(pos, random.choice(FILLER_TOKENS).strip())
    return " ".join(words)


def _perturb_synonym_substitution(text, k):
    words = text.split()
    idxs = [i for i, w in enumerate(words) if w.strip(".,").lower() in SYNONYM_MAP]
    random.shuffle(idxs)
    for i in idxs[:k]:
        w = words[i]
        core = w.strip(".,").lower()
        suffix = w[len(core):]
        words[i] = SYNONYM_MAP.get(core, core) + suffix
    return " ".join(words)


_PERTURB_FUNCS = {
    "abbreviation_swap": _perturb_abbreviation_swap,
    "ocr_noise": _perturb_ocr_noise,
    "token_insertion": _perturb_token_insertion,
    "synonym_substitution": _perturb_synonym_substitution,
}


def sample_perturbation(text, perturbation_space, epsilon_budget):
    ptype = random.choice(perturbation_space)
    k = random.randint(1, epsilon_budget)
    return _PERTURB_FUNCS[ptype](text, k), ptype


def jaccard_similarity(idx_a, idx_b):
    a, b = set(idx_a.tolist() if torch.is_tensor(idx_a) else idx_a), \
           set(idx_b.tolist() if torch.is_tensor(idx_b) else idx_b)
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def clopper_pearson_lower(successes, n, alpha):
    if n == 0:
        return 0.0
    if successes == 0:
        return 0.0
    return float(stats.beta.ppf(alpha, successes, n - successes + 1))


@torch.no_grad()
def certify(model, text, tokenizer, device, perturbation_space=PERTURBATION_SPACE,
            epsilon_budget=2, n_samples=100, alpha=0.01, theta_stability=0.5,
            evidence_threshold=0.5, max_len=128):
    """v6: n_samples default raised. At n_samples=100, alpha=0.01, a
    perfect 100/100 run yields Clopper-Pearson lower bound ~0.955, clearing
    the 0.99 threshold only marginally -- raise further (e.g. 300-500) for
    alpha=0.01 specifically, since alpha=0.01 is a stricter bar than the
    alpha=0.05 used elsewhere. See run_certification_suite()'s printed
    ceiling warning for the exact achievable bound at your chosen settings."""

    def forward_once(t):
        enc = tokenizer(t, truncation=True, max_length=max_len, return_offsets_mapping=True)
        input_ids = torch.tensor([enc["input_ids"]], dtype=torch.long).to(device)
        attn_mask = torch.tensor([enc["attention_mask"]], dtype=torch.long).to(device)
        out = model.gated_forward(input_ids, attn_mask, evidence_threshold, check_necessity=True)[0]
        return out

    base_out = forward_once(text)
    if base_out["abstain"]:
        return {"status": "UNCERTIFIED", "reason": "base_input_abstained",
                "prediction_confidence": 0.0, "provenance_confidence": 0.0,
                "p_pred_lower": 0.0, "p_evi_lower": 0.0}

    y0 = torch.argmax(base_out["probs"]).item()
    E0 = torch.nonzero(base_out["ev_mask"], as_tuple=True)[0]

    pred_matches, evidence_matches = [], []
    perturbation_log = []

    for _ in range(n_samples):
        x_pert, ptype = sample_perturbation(text, perturbation_space, epsilon_budget)
        out = forward_once(x_pert)
        perturbation_log.append(ptype)

        if out["abstain"]:
            pred_matches.append(0)
            evidence_matches.append(0)
            continue

        pred_matches.append(int(torch.argmax(out["probs"]).item() == y0))
        E_pert = torch.nonzero(out["ev_mask"], as_tuple=True)[0]
        sim = jaccard_similarity(E_pert, E0)
        evidence_matches.append(int(sim >= theta_stability))

    n_pred_success = sum(pred_matches)
    n_evi_success = sum(evidence_matches)

    p_pred_lower = clopper_pearson_lower(n_pred_success, n_samples, alpha / 2)
    p_evi_lower = clopper_pearson_lower(n_evi_success, n_samples, alpha / 2)

    certified = (p_pred_lower >= 1 - alpha) and (p_evi_lower >= 1 - alpha)

    result = {
        "status": "CERTIFIED" if certified else "UNCERTIFIED",
        "prediction_confidence": p_pred_lower,
        "provenance_confidence": p_evi_lower,
        "p_pred_lower": p_pred_lower,
        "p_evi_lower": p_evi_lower,
        "n_pred_success": n_pred_success,
        "n_evi_success": n_evi_success,
        "n_samples": n_samples,
        "alpha": alpha,
        "theta_stability": theta_stability,
        "epsilon_budget": epsilon_budget,
        "perturbation_space": perturbation_space,
        "perturbation_counts": dict(Counter(perturbation_log)),
    }
    if not certified:
        result["reason"] = "prediction" if p_pred_lower < 1 - alpha else "provenance"
    return result


def run_certification_suite(model, df_test, tokenizer, device, n_examples=30,
                             n_samples=100, alpha=0.01, epsilon_budget=2, theta_stability=0.5):
    sample_df = df_test.sample(n=min(n_examples, len(df_test)), random_state=0)
    certs = []
    for _, row in sample_df.iterrows():
        cert = certify(model, row["text"], tokenizer, device, n_samples=n_samples,
                        alpha=alpha, epsilon_budget=epsilon_budget, theta_stability=theta_stability)
        cert["note_id"] = row["note_id"]
        certs.append(cert)

    n_certified = sum(1 for c in certs if c["status"] == "CERTIFIED")
    n_base_abstained = sum(1 for c in certs if c.get("reason") == "base_input_abstained")
    summary = {
        "n_examples": len(certs),
        "n_certified": n_certified,
        "n_base_abstained": n_base_abstained,
        "certification_rate": n_certified / len(certs) if certs else 0.0,
        "mean_prediction_confidence": float(np.mean([c["prediction_confidence"] for c in certs])),
        "mean_provenance_confidence": float(np.mean([c["provenance_confidence"] for c in certs])),
        "per_example": certs,
    }
    if n_base_abstained > 0:
        print(f"  [warn] {n_base_abstained}/{len(certs)} certification examples abstained on the "
              f"UNPERTURBED base input -- the model is not producing predictions on clean data "
              f"for these examples, which caps certification_rate regardless of robustness. "
              f"If this is a large fraction, revisit training (--min_coverage / --lambda_evi / "
              f"--lambda_coverage_floor / --epochs) before trusting this certification result.")
    required_ceiling = clopper_pearson_lower(n_samples, n_samples, alpha / 2)
    if required_ceiling < 1 - alpha:
        print(f"  [info] n_samples={n_samples} at alpha={alpha}: the Clopper-Pearson lower bound "
              f"cannot exceed ~{required_ceiling:.3f} even with a perfect success rate, below the "
              f"1-alpha={1-alpha:.2f} threshold required for CERTIFIED status. Raise "
              f"--cert_n_samples if you want a realistic chance of positive certificates.")
    else:
        print(f"  [info] n_samples={n_samples} at alpha={alpha}: a perfect success rate CAN reach "
              f"CERTIFIED status (ceiling={required_ceiling:.3f} >= {1-alpha:.2f}).")
    return summary


# ----------------------------------------------------------------------------
# 7. Main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out_dir", default="output")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--finetune_bert", action="store_true")

    # v6: min_coverage (inference gate) and min_coverage_target (training
    # floor) are now separate. v5 used a single 0.05 value for both.
    parser.add_argument("--min_coverage", type=float, default=0.08,
                         help="Inference-time hard-gate threshold (gated_forward()).")
    parser.add_argument("--min_coverage_target", type=float, default=0.10,
                         help="Training-time floor loss target (train_step()'s L_floor). "
                              "Should be >= --min_coverage to leave margin for dev/test variance.")
    parser.add_argument("--min_coverage_stop_threshold", type=float, default=None,
                         help="If set, training stops early when mean_train_coverage falls below "
                              "this value after --warmup_epochs. Defaults to --min_coverage if unset.")

    parser.add_argument("--lambda_evi", type=float, default=0.0005,
                         help="Sparsity pressure on evidence mask. Lowered from v5's 0.002 default; "
                              "only active after --warmup_epochs (see apply_evi_penalty in train()).")
    parser.add_argument("--lambda_coverage_floor", type=float, default=1.0,
                         help="NEW in v6. Weight on the hinge loss that penalizes coverage falling "
                              "below --min_coverage_target. This is the core fix for the v5 "
                              "collapse: verified numerically to dominate L_evi by ~4600x at "
                              "collapsed coverage (0.03) while going fully silent at healthy "
                              "coverage (0.15).")
    parser.add_argument("--lambda_faith", type=float, default=0.5)
    parser.add_argument("--lambda_causal", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=6,
                         help="Raised from v5's default of 3. During warmup, L_evi is fully zeroed "
                              "(not just present with a small weight) and the hard gate is not "
                              "enforced in dev eval, so a stable coverage level can establish "
                              "before any sparsity pressure or gating is applied.")
    parser.add_argument("--tau", type=float, default=0.5)

    parser.add_argument("--run_certification", action="store_true")
    parser.add_argument("--cert_n_examples", type=int, default=30)
    parser.add_argument("--cert_n_samples", type=int, default=100)
    parser.add_argument("--cert_alpha", type=float, default=0.01)
    parser.add_argument("--cert_epsilon_budget", type=int, default=2)
    parser.add_argument("--cert_theta_stability", type=float, default=0.5)

    parser.add_argument("--site_id", default="bert_site")
    parser.add_argument("--data_dir", default=None,
                         help="If set, load train/dev/test/manip data from CSVs produced by "
                              "synthetic_longitudinal_generator_v1.py in this directory instead "
                              "of calling this script's own generate_datasets(). Expects "
                              "synthetic_notes_v1.csv and (optionally) synthetic_adversarial_v1.csv.")
    args = parser.parse_args()

    if args.min_coverage_stop_threshold is None:
        args.min_coverage_stop_threshold = args.min_coverage

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loss weights: lambda_evi={args.lambda_evi} lambda_coverage_floor={args.lambda_coverage_floor} "
          f"lambda_faith={args.lambda_faith} lambda_causal={args.lambda_causal} (tau={args.tau}, "
          f"min_coverage={args.min_coverage}, min_coverage_target={args.min_coverage_target}, "
          f"warmup_epochs={args.warmup_epochs}, finetune_bert={args.finetune_bert})")

    if args.data_dir:
        print(f"[1/9] Loading external dataset from {args.data_dir} "
              f"(synthetic_longitudinal_generator_v1.py output)...")
        df_train, df_dev, df_test, df_manip = load_external_dataset(args.data_dir, seed=args.seed)
    else:
        print("[1/9] Generating synthetic clinical dataset (order-balanced, v5/v6)...")
        df_train, df_dev, df_test, df_manip = generate_datasets(seed=args.seed, out_dir=args.out_dir)
    print(f"  train={len(df_train)} dev={len(df_dev)} test={len(df_test)} manip_pairs={len(df_manip)}")
    print(f"  mention_order distribution in test set: {df_test['mention_order'].value_counts().to_dict()}")

    print(f"[2/9] Loading tokenizer/model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    label2idx = {"EVENT1-BEFORE-EVENT2": 0, "EVENT2-BEFORE-EVENT1": 1, "EVENT1-OVERLAP-EVENT2": 2}

    train_ds = ClinicalRelationDataset(df_train, tokenizer, label2idx, max_len=args.max_len)
    dev_ds = ClinicalRelationDataset(df_dev, tokenizer, label2idx, max_len=args.max_len)
    test_ds = ClinicalRelationDataset(df_test, tokenizer, label2idx, max_len=args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    print("[3/9] Training baseline (soft, no gate) and provenance-first "
          "(hard gate + necessity + coverage-floor TRAIN_STEP objective) models...")

    baseline_model = ProvenanceFirstBERT(model_name=args.model_name, hard_gate=False,
                                          finetune_bert=args.finetune_bert).to(device)
    train(baseline_model, train_loader, dev_loader, device, epochs=args.epochs, lr=args.lr,
          tau=args.tau, lambda_evi=args.lambda_evi, lambda_faith=args.lambda_faith,
          lambda_causal=args.lambda_causal, lambda_coverage_floor=args.lambda_coverage_floor,
          min_coverage_target=args.min_coverage_target, warmup_epochs=args.warmup_epochs,
          min_coverage_stop_threshold=None)  # baseline has no hard gate; never early-stops

    pf_model = ProvenanceFirstBERT(model_name=args.model_name, hard_gate=True,
                                    min_coverage=args.min_coverage,
                                    necessity_drop_threshold=0.1,
                                    finetune_bert=args.finetune_bert).to(device)
    train(pf_model, train_loader, dev_loader, device, epochs=args.epochs, lr=args.lr,
          tau=args.tau, lambda_evi=args.lambda_evi, lambda_faith=args.lambda_faith,
          lambda_causal=args.lambda_causal, lambda_coverage_floor=args.lambda_coverage_floor,
          min_coverage_target=args.min_coverage_target, warmup_epochs=args.warmup_epochs,
          min_coverage_stop_threshold=args.min_coverage_stop_threshold)

    print("[4/9] Evaluating on held-out test set...")
    baseline_test_metrics = evaluate(baseline_model, test_loader, device, enforce_gate=False)
    pf_test_metrics = evaluate(pf_model, test_loader, device, enforce_gate=True)
    print(f"  baseline test: abstain={baseline_test_metrics['abstain_rate']:.3f} "
          f"acc={baseline_test_metrics['accuracy_on_answered']:.3f} "
          f"mean_coverage={baseline_test_metrics['mean_coverage']:.4f}")
    print(f"  provenance_first test: abstain={pf_test_metrics['abstain_rate']:.3f} "
          f"acc={pf_test_metrics['accuracy_on_answered']:.3f} "
          f"mean_coverage={pf_test_metrics['mean_coverage']:.4f}")
    if pf_test_metrics["abstain_rate"] > 0.5:
        print(f"  [warn] provenance_first model abstains on {pf_test_metrics['abstain_rate']:.1%} of "
              f"test examples. Manipulation and certification results below will be dominated by "
              f"abstentions rather than reflecting gate behavior on answered examples. Consider "
              f"rerunning with --finetune_bert, more --epochs, lower --lambda_evi, or higher "
              f"--lambda_coverage_floor.")

    print("[4b/9] Running position-swap control test (isolates position from content)...")
    if args.data_dir:
        print("  [skip] synthetic_longitudinal_generator_v1.py only emits forward-order admission "
              "sentences (no CUES_*_REVERSE analogue yet), so there is no order-swapped variant to "
              "test against in external-data mode. Treat per_template_abstain_rate above with that "
              "gap in mind, or extend the generator with a reverse-order sentence template before "
              "relying on this check.")
        posctrl_metrics = None
    else:
        df_posctrl = generate_position_control_set(df_test, seed=99)
        posctrl_ds = ClinicalRelationDataset(df_posctrl, tokenizer, label2idx, max_len=args.max_len)
        posctrl_loader = DataLoader(posctrl_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        posctrl_metrics = evaluate(pf_model, posctrl_loader, device, enforce_gate=True)
        print(f"  position-control abstain rate: {posctrl_metrics['abstain_rate']:.3f} "
              f"(compare to normal test abstain rate: {pf_test_metrics['abstain_rate']:.3f}); "
              f"per-template breakdown: {posctrl_metrics['per_template_abstain_rate']}")

    results = {
        "config": vars(args),
        "test_metrics": {"baseline": baseline_test_metrics, "provenance_first": pf_test_metrics},
        "position_control_metrics": posctrl_metrics,
    }

    print("[5/9] Running provenance-manipulation experiment...")
    all_base_lookup = {r["note_id"]: r for _, r in pd.concat([df_train, df_dev, df_test]).iterrows()}
    baseline_manip = run_manipulation_eval(baseline_model, df_manip, all_base_lookup, tokenizer, device)
    pf_manip = run_manipulation_eval(pf_model, df_manip, all_base_lookup, tokenizer, device)
    results["manipulation_metrics"] = {
        "baseline": summarize_manipulation(baseline_manip),
        "provenance_first": summarize_manipulation(pf_manip),
    }
    print(f"  baseline manip: {results['manipulation_metrics']['baseline']}")
    print(f"  provenance_first manip: {results['manipulation_metrics']['provenance_first']}")

    if args.run_certification:
        print(f"[6/9] Running dual robustness / provenance-stability certification (CERTIFY)...")
        results["certification"] = {
            "baseline": run_certification_suite(
                baseline_model, df_test, tokenizer, device, n_examples=args.cert_n_examples,
                n_samples=args.cert_n_samples, alpha=args.cert_alpha,
                epsilon_budget=args.cert_epsilon_budget, theta_stability=args.cert_theta_stability),
            "provenance_first": run_certification_suite(
                pf_model, df_test, tokenizer, device, n_examples=args.cert_n_examples,
                n_samples=args.cert_n_samples, alpha=args.cert_alpha,
                epsilon_budget=args.cert_epsilon_budget, theta_stability=args.cert_theta_stability),
        }
        for name in ["baseline", "provenance_first"]:
            c = results["certification"][name]
            print(f"  {name}: certification_rate={c['certification_rate']:.3f} "
                  f"mean_pred_conf={c['mean_prediction_confidence']:.3f} "
                  f"mean_evi_conf={c['mean_provenance_confidence']:.3f} "
                  f"n_base_abstained={c['n_base_abstained']}/{c['n_examples']}")
    else:
        print("[6/9] Skipping certification (pass --run_certification to enable).")

    print(f"[7/9] Building provenance temporal knowledge graph (site_id={args.site_id})...")
    kg_baseline = build_knowledge_graph(baseline_model, test_loader, device,
                                         site_id=f"{args.site_id}_baseline", model_id="bio_clinicalbert_baseline")
    kg_pf = build_knowledge_graph(pf_model, test_loader, device,
                                   site_id=f"{args.site_id}_provenance_first", model_id="bio_clinicalbert_pf")
    summary_baseline = aggregate_temporal_motifs(kg_baseline)
    summary_pf = aggregate_temporal_motifs(kg_pf)
    print(f"  [kg] baseline site summary: {summary_baseline}")
    print(f"  [kg] provenance_first site summary: {summary_pf}")
    results["knowledge_graph_summary"] = {"baseline": summary_baseline, "provenance_first": summary_pf}

    with open(os.path.join(args.out_dir, "provenance_kg_baseline.pkl"), "wb") as f:
        pickle.dump(kg_baseline, f)
    with open(os.path.join(args.out_dir, "provenance_kg_provenance_first.pkl"), "wb") as f:
        pickle.dump(kg_pf, f)
    print(f"  [kg] graphs pickled to {args.out_dir}/provenance_kg_baseline.pkl and "
          f"{args.out_dir}/provenance_kg_provenance_first.pkl")

    print("[8/9] Writing results...")
    out_path = os.path.join(args.out_dir, "poc_results_clinicalbert_v6.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n[9/9] Done. Results written to {out_path}")
    print("Before trusting any number above: check mean_train_coverage in the final training "
          "epoch (should stabilize ~0.10-0.30, not slide toward 0.02-0.05), "
          "test_metrics.provenance_first.shortcut_flag, and compare "
          "position_control_metrics.abstain_rate to the normal test abstain rate.")


if __name__ == "__main__":
    main()
