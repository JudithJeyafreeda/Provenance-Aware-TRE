#!/usr/bin/env python3
"""Train/evaluate BioClinicalBERT and export prediction-level CSVs.

"""

import argparse
import json
import os
import random
from typing import Dict, List

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


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def locate_span(text, span):
    if not isinstance(span, str) or not span:
        return -1, -1
    i = text.find(span)
    return (i, i + len(span)) if i >= 0 else (-1, -1)


def evidence_labels(text, span, tokenizer, max_len):
    enc = tokenizer(text, truncation=True, max_length=max_len,
                    return_offsets_mapping=True)
    start, end = locate_span(text, span)
    labels = []
    for s, e in enc["offset_mapping"]:
        labels.append(int(s != e and start >= 0 and s >= start and e <= end))
    return enc["input_ids"], enc["attention_mask"], labels


class RelationDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        ids, mask, ev = evidence_labels(
            r["text"], r["gold_evidence_span"], self.tokenizer, self.max_len
        )
        return {
            "input_ids": ids,
            "attention_mask": mask,
            "evidence_labels": ev,
            "label": LABELS[r["gold_relation"]],
            "example_id": r["note_id"],
            "gold_relation": r["gold_relation"],
        }


def collate(batch, pad_id):
    m = max(len(x["input_ids"]) for x in batch)
    return {
        "input_ids": torch.tensor([
            x["input_ids"] + [pad_id] * (m - len(x["input_ids"])) for x in batch
        ]),
        "attention_mask": torch.tensor([
            x["attention_mask"] + [0] * (m - len(x["attention_mask"])) for x in batch
        ]),
        "evidence_labels": torch.tensor([
            x["evidence_labels"] + [0] * (m - len(x["evidence_labels"])) for x in batch
        ], dtype=torch.float),
        "label": torch.tensor([x["label"] for x in batch]),
        "example_ids": [x["example_id"] for x in batch],
        "gold_relations": [x["gold_relation"] for x in batch],
    }


class EvidenceBERT(nn.Module):
    def __init__(self, model_name, gated, min_coverage, min_target, finetune):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        h = self.encoder.config.hidden_size
        if not finetune:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.evidence_head = nn.Linear(h, 1)
        self.relation_head = nn.Linear(h, 3)
        self.gated = gated
        self.min_coverage = min_coverage
        self.min_target = min_target

    def forward_features(self, ids, mask):
        hidden = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        scores = torch.sigmoid(self.evidence_head(hidden).squeeze(-1))
        scores = scores.masked_fill(mask == 0, 0.0)
        return hidden, scores

    def train_forward(self, ids, mask):
        hidden, scores = self.forward_features(ids, mask)
        weights = scores * mask.float()
        pooled = (weights.unsqueeze(-1) * hidden).sum(1) / weights.sum(1, keepdim=True).clamp_min(1e-8)
        return pooled, scores

    @torch.no_grad()
    def predict(self, ids, mask, evidence_threshold, enforce_gate=True):
        hidden, scores = self.forward_features(ids, mask)
        outputs = []
        for i in range(ids.size(0)):
            valid = mask[i].bool()
            h = hidden[i][valid]
            s = scores[i][valid]
            ev_mask = (s >= evidence_threshold).float()
            coverage = float(ev_mask.mean().item()) if len(ev_mask) else 0.0
            reason = None
            abstain = False
            if self.gated and enforce_gate and coverage < self.min_coverage:
                abstain = True
                reason = "low_coverage"
            if abstain:
                outputs.append({
                    "abstained": True, "answer": None, "coverage": coverage,
                    "evidence_score": float(s.mean().item()) if len(s) else 0.0,
                    "gate_reason": reason, "ev_mask": ev_mask,
                })
                continue
            w = (ev_mask * s) if self.gated else s
            w = w / w.sum().clamp_min(1e-8)
            pooled = (w.unsqueeze(-1) * h).sum(0)
            probs = F.softmax(self.relation_head(pooled), dim=-1)
            outputs.append({
                "abstained": False, "answer": int(torch.argmax(probs).item()),
                "coverage": coverage,
                "evidence_score": float(s.mean().item()) if len(s) else 0.0,
                "gate_reason": None, "ev_mask": ev_mask,
            })
        return outputs


def train_one(model, loader, device, epochs, lr, gold_weight, floor_weight):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    steps = 0
    for epoch in range(epochs):
        model.train()
        losses = []
        for b in loader:
            ids, mask = b["input_ids"].to(device), b["attention_mask"].to(device)
            labels = b["label"].to(device)
            gold = b["evidence_labels"].to(device)
            pooled, scores = model.train_forward(ids, mask)
            logits = model.relation_head(pooled)
            valid = mask.bool()
            gold_losses = []
            floor_losses = []
            for i in range(ids.size(0)):
                s = scores[i][valid[i]]
                g = gold[i][valid[i]]
                gold_losses.append(F.binary_cross_entropy(s, g))
                soft_cov = s.mean()
                floor_losses.append(F.relu(torch.tensor(model.min_target, device=device) - soft_cov))
            loss = (
                F.cross_entropy(logits, labels)
                + gold_weight * torch.stack(gold_losses).mean()
                + floor_weight * torch.stack(floor_losses).mean()
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            losses.append(float(loss.item()))
            steps += 1
        print(f"epoch={epoch + 1}/{epochs} loss={np.mean(losses):.4f}")
    return steps


def export_predictions(model, loader, df, device, model_name, threshold, out_path,
                       enforce_gate=True):
    model.eval()
    gold_by_id = df.set_index("note_id")["gold_relation"].to_dict()
    rows = []
    with torch.no_grad():
        for b in loader:
            ids, mask = b["input_ids"].to(device), b["attention_mask"].to(device)
            outputs = model.predict(ids, mask, threshold, enforce_gate=enforce_gate)
            for example_id, out in zip(b["example_ids"], outputs):
                answer = out["answer"]
                gold = LABELS[gold_by_id[example_id]]
                rows.append({
                    "example_id": example_id,
                    "model": model_name,
                    "answer": answer,
                    "abstained": bool(out["abstained"]),
                    "correct": None if out["abstained"] else int(answer == gold),
                    "evidence_score": out["evidence_score"],
                    "evidence_sufficient": None,
                    "contradiction_present": None,
                    "provenance_recorded": True,
                    "gate_reason": out["gate_reason"],
                    "coverage": out["coverage"],
                    "evidence_recall": None,
                    "evidence_span_valid": None,
                })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"wrote {len(rows)} rows to {out_path}")


def load_data(data_dir, seed):
    notes = pd.read_csv(os.path.join(data_dir, "synthetic_notes_v1.csv"))
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", default="bert_predictions")
    p.add_argument("--model_name", default="emilyalsentzer/Bio_ClinicalBERT")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--evidence_threshold", type=float, default=0.5)
    p.add_argument("--min_coverage", type=float, default=0.08)
    p.add_argument("--min_target", type=float, default=0.10)
    p.add_argument("--gold_weight", type=float, default=0.3)
    p.add_argument("--floor_weight", type=float, default=1.0)
    p.add_argument("--finetune", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_df, dev_df, test_df = load_data(args.data_dir, args.seed)
    train_ds = RelationDataset(train_df, tokenizer, args.max_len)
    dev_ds = RelationDataset(dev_df, tokenizer, args.max_len)
    test_ds = RelationDataset(test_df, tokenizer, args.max_len)
    cf = lambda b: collate(b, tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, collate_fn=cf)
    dev_loader = DataLoader(dev_ds, args.batch_size, shuffle=False, collate_fn=cf)
    test_loader = DataLoader(test_ds, args.batch_size, shuffle=False, collate_fn=cf)

    baseline = EvidenceBERT(args.model_name, False, args.min_coverage, args.min_target, args.finetune).to(device)
    gated = EvidenceBERT(args.model_name, True, args.min_coverage, args.min_target, args.finetune).to(device)
    print("training baseline")
    b_steps = train_one(baseline, train_loader, device, args.epochs, args.lr, args.gold_weight, args.floor_weight)
    print("training provenance-first")
    g_steps = train_one(gated, train_loader, device, args.epochs, args.lr, args.gold_weight, args.floor_weight)
    assert b_steps > 0 and g_steps > 0

    for split, loader, df in [("dev", dev_loader, dev_df), ("test", test_loader, test_df)]:
        export_predictions(baseline, loader, df, device, "bert_baseline", args.evidence_threshold,
                           os.path.join(args.out_dir, f"bert_baseline_{split}_predictions.csv"), False)
        export_predictions(gated, loader, df, device, "bert_provenance_first", args.evidence_threshold,
                           os.path.join(args.out_dir, f"bert_provenance_first_{split}_predictions.csv"), True)

    with open(os.path.join(args.out_dir, "bert_run_metadata.json"), "w") as f:
        json.dump({"config": vars(args), "device": str(device), "train_steps_baseline": b_steps,
                   "train_steps_provenance_first": g_steps, "train_n": len(train_df),
                   "dev_n": len(dev_df), "test_n": len(test_df)}, f, indent=2)


if __name__ == "__main__":
    main()
