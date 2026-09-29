#!/usr/bin/env python3
"""

Usage (weights already saved from v2's --save_dir):
python train_availability_supervised_bert_v3.py \
  --data_dir synthetic_dataset --out_dir availability_supervised_bert_v2 \
  --load_dir availability_supervised_bert_v2/weights \
  --export_predicted_spans --gold_spans_v2 gold_spans_v2/gold_spans_v2.csv

Can be combined with --extra_controls on the same call. For silent_removal in
particular, read the control's own abstention/accuracy numbers as the main
result, not evidence_recall_v2: that control's whole point is that evidence is
gone, so a low or null recall there is expected, not a defect.
"""


from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer

LABELS = {
    "EVENT1-BEFORE-EVENT2": 0,
    "EVENT2-BEFORE-EVENT1": 1,
    "EVENT1-OVERLAP-EVENT2": 2,
}
INV_LABELS = {v: k for k, v in LABELS.items()}


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def remove_evidence_text(text: str, gold_span: str) -> str:
    replacement = "The note does not document the temporal relationship between the two events."
    if isinstance(gold_span, str) and gold_span and gold_span in text:
        return text.replace(gold_span, replacement, 1)
    return (
        "The patient was admitted for evaluation. "
        "The available documentation does not specify the temporal relationship "
        "between the two clinical events."
    )


def mask_temporal_text(text: str) -> str:
    cues = [
        "prior to", "before", "preceding", "following", "after",
        "subsequent to", "during", "concurrent with", "while undergoing",
    ]
    out = text
    for cue in cues:
        if cue.lower() in out.lower():
            start = out.lower().find(cue.lower())
            return out[:start] + "[MASKED_TEMPORAL_CUE]" + out[start + len(cue):]
    return out + " The temporal cue needed to determine the relation was omitted."


def make_unavailable_controls(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    out = df.copy()
    if variant == "remove_evidence":
        out["text"] = [remove_evidence_text(t, s) for t, s in zip(out.text, out.gold_evidence_span)]
    elif variant == "mask_temporal_cue":
        out["text"] = out["text"].map(mask_temporal_text)
    else:
        raise ValueError(variant)
    out["note_id"] = out["note_id"].astype(str) + "_" + variant
    out["evidence_available"] = 0
    out["evidence_case"] = variant
    out["gold_evidence_span"] = ""
    return out


def load_clean_relation_notes(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "synthetic_notes_v1.csv")
    df = pd.read_csv(path)
    df = df[df["gold_relation"].notna()].copy()
    df = df[df["gold_evidence_span"].notna()].copy()
    df["evidence_available"] = 1
    df["evidence_case"] = "clean_evidence_available"
    return df.reset_index(drop=True)


def split_by_patient(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    patients = list(df["patient_id"].unique())
    random.Random(seed).shuffle(patients)
    n_test = max(1, int(0.15 * len(patients)))
    n_dev = max(1, int(0.15 * len(patients)))
    test_p = set(patients[:n_test])
    dev_p = set(patients[n_test:n_test + n_dev])
    train_p = set(patients[n_test + n_dev:])
    return (
        df[df.patient_id.isin(train_p)].reset_index(drop=True),
        df[df.patient_id.isin(dev_p)].reset_index(drop=True),
        df[df.patient_id.isin(test_p)].reset_index(drop=True),
    )


def build_control_splits(clean_train, clean_dev, clean_test):
    def add_controls(df):
        return pd.concat([
            df,
            make_unavailable_controls(df, "remove_evidence"),
            make_unavailable_controls(df, "mask_temporal_cue"),
        ], ignore_index=True)
    return add_controls(clean_train), add_controls(clean_dev), add_controls(clean_test)


class DatasetRows(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        enc = self.tokenizer(
            str(r.text), truncation=True, max_length=self.max_len,
            return_offsets_mapping=True,
        )
        span = str(r.gold_evidence_span) if pd.notna(r.gold_evidence_span) else ""
        start = str(r.text).find(span) if span else -1
        end = start + len(span) if start >= 0 else -1
        evidence = [int(s != e and start >= 0 and s >= start and e <= end)
                    for s, e in enc["offset_mapping"]]
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "evidence_labels": evidence,
            "relation": LABELS[r.gold_relation],
            "availability": int(r.evidence_available),
            "example_id": str(r.note_id),
            "gold_relation": r.gold_relation,
            "evidence_case": r.evidence_case,
        }


def prepare_extra_controls(path: str, test_patients) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"note_id", "patient_id", "text", "gold_relation"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df = df[df["patient_id"].isin(test_patients)].copy()
    if "evidence_available_oracle" in df.columns:
        df["evidence_available"] = df["evidence_available_oracle"].astype(int)
    else:
        df["evidence_available"] = 0
    df["gold_evidence_span"] = ""
    if "evidence_case" not in df.columns:
        df["evidence_case"] = os.path.splitext(os.path.basename(path))[0]
    df["note_id"] = df["note_id"].astype(str)
    return df.reset_index(drop=True)


def head_state(model, finetune: bool):
    sd = model.state_dict()
    return sd if finetune else {k: v for k, v in sd.items() if not k.startswith("encoder.")}


def collate(batch, pad_id):
    m = max(len(x["input_ids"]) for x in batch)
    return {
        "input_ids": torch.tensor([x["input_ids"] + [pad_id] * (m - len(x["input_ids"])) for x in batch]),
        "attention_mask": torch.tensor([x["attention_mask"] + [0] * (m - len(x["attention_mask"])) for x in batch]),
        "evidence_labels": torch.tensor([x["evidence_labels"] + [0] * (m - len(x["evidence_labels"])) for x in batch], dtype=torch.float),
        "relation": torch.tensor([x["relation"] for x in batch]),
        "availability": torch.tensor([x["availability"] for x in batch], dtype=torch.float),
        "example_ids": [x["example_id"] for x in batch],
        "gold_relations": [x["gold_relation"] for x in batch],
        "evidence_cases": [x["evidence_case"] for x in batch],
    }


class AvailabilityBERT(nn.Module):
    def __init__(self, model_name, provenance_first, finetune=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        h = self.encoder.config.hidden_size
        if not finetune:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.evidence_head = nn.Linear(h, 1)
        self.relation_head = nn.Linear(h, 3)
        self.availability_head = nn.Linear(h, 1)
        self.provenance_first = provenance_first

    def encode(self, ids, mask):
        hidden = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        evidence_scores = torch.sigmoid(self.evidence_head(hidden).squeeze(-1))
        evidence_scores = evidence_scores.masked_fill(mask == 0, 0.0)
        weights = evidence_scores * mask.float()
        pooled = (weights.unsqueeze(-1) * hidden).sum(1) / weights.sum(1, keepdim=True).clamp_min(1e-8)
        availability_logits = self.availability_head(pooled).squeeze(-1)
        relation_logits = self.relation_head(pooled)
        return hidden, evidence_scores, pooled, relation_logits, availability_logits

    @torch.no_grad()
    def predict(self, ids, mask, availability_threshold):
        _, evidence, pooled, relation_logits, availability_logits = self.encode(ids, mask)
        availability = torch.sigmoid(availability_logits)
        relation_probs = F.softmax(relation_logits, dim=-1)
        outputs = []
        for i in range(ids.size(0)):
            coverage = float((evidence[i][mask[i].bool()] >= 0.1).float().mean().item())
            avail = float(availability[i].item())
            abstain = self.provenance_first and avail < availability_threshold
            outputs.append({
                "abstained": bool(abstain),
                "answer": None if abstain else int(torch.argmax(relation_probs[i]).item()),
                "availability_probability": avail,
                "evidence_score": float(evidence[i][mask[i].bool()].mean().item()),
                "coverage": coverage,
                "gate_reason": "evidence_unavailable" if abstain else None,
            })
        return outputs


def train(model, loader, device, epochs, lr, lambda_availability, lambda_evidence):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)
    steps = 0
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            relation = batch["relation"].to(device)
            availability = batch["availability"].to(device)
            gold_evidence = batch["evidence_labels"].to(device)
            _, evidence, _, relation_logits, availability_logits = model.encode(ids, mask)
            relation_loss = F.cross_entropy(relation_logits, relation)
            valid = mask.bool()
            e_losses = []
            for i in range(ids.size(0)):
                e_losses.append(F.binary_cross_entropy(evidence[i][valid[i]], gold_evidence[i][valid[i]]))
            evidence_loss = torch.stack(e_losses).mean()
            availability_loss = F.binary_cross_entropy_with_logits(availability_logits, availability)
            loss = relation_loss + lambda_evidence * evidence_loss
            if model.provenance_first:
                loss = loss + lambda_availability * availability_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
            steps += 1
        print(f"epoch={epoch + 1}/{epochs} loss={np.mean(losses):.4f}")
    return steps


def export(model, loader, df, device, model_name, threshold, output_path):
    model.eval()
    gold_map = df.set_index("note_id")["gold_relation"].to_dict()
    rows = []
    with torch.no_grad():
        for batch in loader:
            outputs = model.predict(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                threshold,
            )
            for example_id, out in zip(batch["example_ids"], outputs):
                answer = INV_LABELS[out["answer"]] if out["answer"] is not None else None
                gold = gold_map[example_id]
                rows.append({
                    "example_id": example_id,
                    "model": model_name,
                    "answer": answer,
                    "abstained": out["abstained"],
                    "correct": None if out["abstained"] else int(answer == gold),
                    "evidence_score": out["evidence_score"],
                    "availability_probability": out["availability_probability"],
                    "evidence_sufficient": None,
                    "contradiction_present": None,
                    "provenance_recorded": True,
                    "gate_reason": out["gate_reason"],
                    "coverage": out["coverage"],
                    "evidence_case": df.set_index("note_id").loc[example_id].get("evidence_case", None),
                })
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"wrote {len(rows)} rows to {output_path}")


def evidence_labels_from_span(text, span, offsets):
    """Same alignment DatasetRows.__getitem__ uses to build gold_evidence labels,
    factored out so it can be reused for both gold (v2) and predicted spans."""
    if not isinstance(span, str) or not span:
        return [0] * len(offsets)
    start = text.find(span)
    if start < 0:
        return [0] * len(offsets)
    end = start + len(span)
    return [int(s != e and s >= start and e <= end) for s, e in offsets]


def span_text_from_token_mask(text, offsets, flags):
    """Inverse of the above: longest contiguous run of flagged tokens -> substring."""
    best = None
    run_start = None
    for idx, (flag, (s, e)) in enumerate(zip(flags, offsets)):
        if flag and s != e:
            if run_start is None:
                run_start = s
            run_end = e
        else:
            if run_start is not None:
                if best is None or (run_end - run_start) > (best[1] - best[0]):
                    best = (run_start, run_end)
                run_start = None
    if run_start is not None and (best is None or (run_end - run_start) > (best[1] - best[0])):
        best = (run_start, run_end)
    return text[best[0]:best[1]] if best else None


def export_with_spans(model, df, tokenizer, device, model_name, availability_threshold,
                      evidence_token_threshold, max_len, output_path, gold_span_v2_map=None):
    """Per-row export (no DataLoader/collate) so offset_mapping is available for both
    reconstructing the predicted span and, if gold_span_v2_map is given, recomputing
    recall against corrected gold spans -- all from a single forward pass per row,
    using whatever weights are already loaded into `model` (no retraining)."""
    model.eval()
    rows = []
    with torch.no_grad():
        for _, r in df.iterrows():
            text = str(r["text"])
            enc = tokenizer(text, truncation=True, max_length=max_len, return_offsets_mapping=True,
                            return_tensors="pt")
            offsets = enc.pop("offset_mapping")[0].tolist()
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            _, evidence, pooled, relation_logits, availability_logits = model.encode(ids, mask)
            avail = float(torch.sigmoid(availability_logits)[0].item())
            relation_probs = torch.softmax(relation_logits, dim=-1)[0]
            m = mask[0].bool()
            ev = evidence[0][m]
            coverage = float((ev >= 0.1).float().mean().item())
            abstain = bool(model.provenance_first and avail < availability_threshold)
            answer = None if abstain else INV_LABELS[int(torch.argmax(relation_probs).item())]
            gold = r["gold_relation"]
            flags = (evidence[0] >= evidence_token_threshold).int().tolist()
            flags = [f for f, valid in zip(flags, mask[0].tolist()) if valid]
            offsets_valid = [o for o, valid in zip(offsets, mask[0].tolist()) if valid]
            pred_span = span_text_from_token_mask(text, offsets_valid, flags)

            row = {
                "example_id": str(r["note_id"]), "model": model_name, "answer": answer,
                "abstained": abstain, "correct": None if abstain else int(answer == gold),
                "evidence_score": float(ev.mean().item()) if len(ev) else 0.0,
                "availability_probability": avail, "coverage": coverage,
                "gate_reason": "evidence_unavailable" if abstain else None,
                "evidence_case": r.get("evidence_case", None),
                "predicted_evidence_span": pred_span,
            }
            if gold_span_v2_map is not None:
                v2 = gold_span_v2_map.get(str(r["note_id"]))
                if v2 and v2.get("reason") == "accepted" and isinstance(v2.get("gold_evidence_span_v2"), str):
                    gold_flags = evidence_labels_from_span(text, v2["gold_evidence_span_v2"], offsets_valid)
                    gold_n = sum(gold_flags)
                    if gold_n > 0:
                        overlap = sum(p * g for p, g in zip(flags, gold_flags))
                        row["evidence_recall_v2"] = overlap / gold_n
                        row["gold_evidence_span_v2_tier"] = v2.get("validity_tier")
                    else:
                        row["evidence_recall_v2"] = None
                        row["gold_evidence_span_v2_tier"] = "gold_span_not_in_truncated_note"
                else:
                    row["evidence_recall_v2"] = None
                    row["gold_evidence_span_v2_tier"] = v2.get("reason") if v2 else "no_v2_row"
            rows.append(row)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False)
    n_recall = int(out_df["evidence_recall_v2"].notna().sum()) if "evidence_recall_v2" in out_df else 0
    mean_recall = out_df["evidence_recall_v2"].mean() if n_recall else None
    if gold_span_v2_map is None:
        extra = ""
    elif mean_recall is None:
        extra = f" | evidence_recall_v2: 0/{len(rows)} scorable rows (no gold_spans_v2 match by note_id for this file)"
    else:
        extra = f" | evidence_recall_v2: mean={mean_recall:.4f} over {n_recall}/{len(rows)} scorable rows"
    print(f"wrote {len(rows)} rows to {output_path}" + extra)
    return out_df


def load_gold_spans_v2(path):
    df = pd.read_csv(path)
    df["note_id"] = df["note_id"].astype(str)
    return {r["note_id"]: r.to_dict() for _, r in df.iterrows()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", default="availability_supervised_bert")
    p.add_argument("--model_name", default="emilyalsentzer/Bio_ClinicalBERT")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--availability_threshold", type=float, default=0.5)
    p.add_argument("--lambda_availability", type=float, default=1.0)
    p.add_argument("--lambda_evidence", type=float, default=0.3)
    p.add_argument("--finetune", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--save_dir", default=None, help="save trained weights here")
    p.add_argument("--load_dir", default=None, help="load weights from here and skip training")
    p.add_argument("--extra_controls", default=None, help="comma-separated control CSVs to score on the test patients")
    p.add_argument("--export_predicted_spans", action="store_true",
                   help="also reconstruct each prediction's evidence span as text")
    p.add_argument("--gold_spans_v2", default=None,
                   help="gold_spans_v2.csv from regenerate_gold_spans.py; with --export_predicted_spans, "
                        "also computes evidence_recall_v2 against these corrected spans")
    p.add_argument("--evidence_token_threshold", type=float, default=0.1,
                   help="per-token evidence-score cutoff for the reconstructed span and evidence_recall_v2")
    args = p.parse_args()
    gold_span_v2_map = load_gold_spans_v2(args.gold_spans_v2) if args.gold_spans_v2 else None
    os.makedirs(args.out_dir, exist_ok=True)
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    clean = load_clean_relation_notes(args.data_dir)
    train_clean, dev_clean, test_clean = split_by_patient(clean, args.seed)
    train_df, dev_df, test_df = build_control_splits(train_clean, dev_clean, test_clean)

    train_loader = DataLoader(
        DatasetRows(train_df, tokenizer, args.max_len),
        args.batch_size, shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )
    dev_loader = DataLoader(
        DatasetRows(dev_df, tokenizer, args.max_len),
        args.batch_size, shuffle=False,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )
    test_loader = DataLoader(
        DatasetRows(test_df, tokenizer, args.max_len),
        args.batch_size, shuffle=False,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )

    regular = AvailabilityBERT(args.model_name, provenance_first=False, finetune=args.finetune).to(device)
    provenance = AvailabilityBERT(args.model_name, provenance_first=True, finetune=args.finetune).to(device)

    names = {"regular_bert": regular, "provenance_first_availability_supervised": provenance}
    if args.load_dir:
        for name, model in names.items():
            state = torch.load(os.path.join(args.load_dir, f"{name}_heads.pt"), map_location=device)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if unexpected:
                raise RuntimeError(f"unexpected keys when loading {name}: {unexpected[:5]}")
            print(f"loaded {name} from {args.load_dir}")
    else:
        print("training regular BERT")
        regular_steps = train(regular, train_loader, device, args.epochs, args.lr, args.lambda_availability, args.lambda_evidence)
        print("training availability-supervised provenance-first BERT")
        provenance_steps = train(provenance, train_loader, device, args.epochs, args.lr, args.lambda_availability, args.lambda_evidence)
        if regular_steps == 0 or provenance_steps == 0:
            raise RuntimeError("No optimizer steps executed.")
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            for name, model in names.items():
                torch.save(head_state(model, args.finetune), os.path.join(args.save_dir, f"{name}_heads.pt"))
            print(f"saved weights to {args.save_dir}")

    clean_test = test_df[test_df.evidence_available.eq(1)].copy()
    unavailable_test = test_df[test_df.evidence_available.eq(0)].copy()
    clean_loader = DataLoader(DatasetRows(clean_test, tokenizer, args.max_len), args.batch_size, shuffle=False, collate_fn=lambda b: collate(b, tokenizer.pad_token_id))
    unavailable_loader = DataLoader(DatasetRows(unavailable_test, tokenizer, args.max_len), args.batch_size, shuffle=False, collate_fn=lambda b: collate(b, tokenizer.pad_token_id))

    for model, name in [(regular, "regular_bert"), (provenance, "provenance_first_availability_supervised")]:
        if args.export_predicted_spans:
            export_with_spans(model, clean_test, tokenizer, device, name, args.availability_threshold,
                              args.evidence_token_threshold, args.max_len,
                              os.path.join(args.out_dir, f"{name}_clean_test_predictions.csv"), gold_span_v2_map)
            export_with_spans(model, unavailable_test, tokenizer, device, name, args.availability_threshold,
                              args.evidence_token_threshold, args.max_len,
                              os.path.join(args.out_dir, f"{name}_unavailable_test_predictions.csv"), gold_span_v2_map)
        else:
            export(model, clean_loader, clean_test, device, name, args.availability_threshold, os.path.join(args.out_dir, f"{name}_clean_test_predictions.csv"))
            export(model, unavailable_loader, unavailable_test, device, name, args.availability_threshold, os.path.join(args.out_dir, f"{name}_unavailable_test_predictions.csv"))

    if args.extra_controls:
        test_patients = set(test_clean.patient_id)
        for path in [x.strip() for x in args.extra_controls.split(",") if x.strip()]:
            extra = prepare_extra_controls(path, test_patients)
            stem = os.path.splitext(os.path.basename(path))[0]
            print(f"{stem}: {len(extra)} rows from {extra.patient_id.nunique()} test patients")
            if not len(extra):
                continue
            if args.export_predicted_spans:
                for name, model in names.items():
                    export_with_spans(model, extra, tokenizer, device, name, args.availability_threshold,
                                      args.evidence_token_threshold, args.max_len,
                                      os.path.join(args.out_dir, f"{name}_{stem}_test_predictions.csv"),
                                      gold_span_v2_map)
            else:
                extra_loader = DataLoader(DatasetRows(extra, tokenizer, args.max_len), args.batch_size, shuffle=False,
                                          collate_fn=lambda b: collate(b, tokenizer.pad_token_id))
                for name, model in names.items():
                    export(model, extra_loader, extra, device, name, args.availability_threshold,
                           os.path.join(args.out_dir, f"{name}_{stem}_test_predictions.csv"))

    with open(os.path.join(args.out_dir, "availability_supervised_metadata.json"), "w") as f:
        json.dump({"config": vars(args), "device": str(device), "train_n": len(train_df), "dev_n": len(dev_df), "test_n": len(test_df)}, f, indent=2)


if __name__ == "__main__":
    main()
