#!/usr/bin/env python3
"""Train BioClinicalBERT on clean notes and/or run inference on a control CSV.
Train: CUDA_VISIBLE_DEVICES="" python bert_control_inference_v1.py \
  --data_dir /mnt/eds_projets/eds_iam/Judith_ARIPPA/work/LLM/TEMPORALEX/output/synthetic_dataset \
  --out_dir /mnt/eds_projets/eds_iam/Judith_ARIPPA/work/LLM/TEMPORALEX/output/synthetic_dataset/BERT_control \
  --epochs 20
Test: CUDA_VISIBLE_DEVICES="" python bert_control_inference_v1.py \
  --data_dir /mnt/eds_projets/eds_iam/Judith_ARIPPA/work/LLM/TEMPORALEX/output/synthetic_dataset \
  --out_dir /mnt/eds_projets/eds_iam/Judith_ARIPPA/work/LLM/TEMPORALEX/output/synthetic_dataset/BERT_control_threshold_test \
  --eval_csv /mnt/eds_projets/eds_iam/Judith_ARIPPA/work/LLM/TEMPORALEX/output/synthetic_dataset/synthetic_notes_v1.csv \
  --eval_only \
  --checkpoint /mnt/eds_projets/eds_iam/Judith_ARIPPA/work/LLM/TEMPORALEX/output/synthetic_dataset/BERT_control/bert_provenance_first_state.pt \
  --min_coverage 0.0 \
  --evidence_threshold 0.5
  """

import argparse
import json
import os
import random

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


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def locate_span(text, span):
    if not isinstance(span, str) or not span.strip():
        return -1, -1
    i = text.find(span)
    return (i, i + len(span)) if i >= 0 else (-1, -1)


def make_example(row, tokenizer, max_len, include_evidence_labels=True):
    text = str(row["text"])
    span = str(row.get("gold_evidence_span", "")) if include_evidence_labels else ""
    enc = tokenizer(text, truncation=True, max_length=max_len, return_offsets_mapping=True)
    start, end = locate_span(text, span)
    ev = []
    for s, e in enc["offset_mapping"]:
        ev.append(int(s != e and start >= 0 and s >= start and e <= end))
    relation = row.get("gold_relation")
    if relation not in LABELS:
        raise ValueError(f"Missing/invalid gold_relation for {row['note_id']!r}: {relation!r}")
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "evidence_labels": ev,
        "label": LABELS[relation],
        "note_id": str(row["note_id"]),
        "gold_relation": relation,
    }


class RelationDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, include_evidence_labels=True):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.include_evidence_labels = include_evidence_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        return make_example(
            self.df.iloc[i], self.tokenizer, self.max_len,
            self.include_evidence_labels,
        )


def collate(batch, pad_id):
    m = max(len(x["input_ids"]) for x in batch)
    return {
        "input_ids": torch.tensor([
            x["input_ids"] + [pad_id] * (m - len(x["input_ids"])) for x in batch
        ], dtype=torch.long),
        "attention_mask": torch.tensor([
            x["attention_mask"] + [0] * (m - len(x["attention_mask"])) for x in batch
        ], dtype=torch.long),
        "evidence_labels": torch.tensor([
            x["evidence_labels"] + [0] * (m - len(x["evidence_labels"])) for x in batch
        ], dtype=torch.float),
        "labels": torch.tensor([x["label"] for x in batch], dtype=torch.long),
        "note_ids": [x["note_id"] for x in batch],
        "gold_relations": [x["gold_relation"] for x in batch],
    }


class ProvenanceFirstBERT(nn.Module):
    def __init__(self, model_name, gated, min_coverage, min_target, finetune=False):
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

    def features(self, ids, mask):
        hidden = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        scores = torch.sigmoid(self.evidence_head(hidden).squeeze(-1))
        scores = scores.masked_fill(mask == 0, 0.0)
        return hidden, scores

    def train_forward(self, ids, mask):
        hidden, scores = self.features(ids, mask)
        weights = scores * mask.float()
        pooled = (weights.unsqueeze(-1) * hidden).sum(1) / weights.sum(1, keepdim=True).clamp_min(1e-8)
        return pooled, scores

    @torch.no_grad()
    def predict(self, ids, mask, evidence_threshold, enforce_gate=True):
        hidden, scores = self.features(ids, mask)
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
                    "abstained": True,
                    "answer": None,
                    "coverage": coverage,
                    "evidence_score": float(s.mean().item()) if len(s) else 0.0,
                    "gate_reason": reason,
                })
                continue

            weights = ev_mask * s if self.gated else s
            weights = weights / weights.sum().clamp_min(1e-8)
            pooled = (weights.unsqueeze(-1) * h).sum(0)
            probs = F.softmax(self.relation_head(pooled), dim=-1)
            outputs.append({
                "abstained": False,
                "answer": int(torch.argmax(probs).item()),
                "coverage": coverage,
                "evidence_score": float(s.mean().item()) if len(s) else 0.0,
                "gate_reason": None,
            })
        return outputs


def train_model(model, loader, device, epochs, lr, gold_weight, floor_weight):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)
    steps = 0
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            gold = batch["evidence_labels"].to(device)
            pooled, scores = model.train_forward(ids, mask)
            logits = model.relation_head(pooled)
            gold_losses = []
            floor_losses = []
            for i in range(ids.size(0)):
                s = scores[i][mask[i].bool()]
                g = gold[i][mask[i].bool()]
                gold_losses.append(F.binary_cross_entropy(s, g))
                floor_losses.append(F.relu(model.min_target - s.mean()))
            loss = (
                F.cross_entropy(logits, labels)
                + gold_weight * torch.stack(gold_losses).mean()
                + floor_weight * torch.stack(floor_losses).mean()
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
            steps += 1
        print(f"epoch={epoch + 1}/{epochs} loss={np.mean(losses):.4f}")
    return steps


def load_clean_splits(data_dir, seed):
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


def load_control_csv(path):
    df = pd.read_csv(path)

    required = {"note_id", "text", "gold_relation"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Control CSV missing columns: {sorted(missing)}")

    df = df[df["gold_relation"].notna()].copy()

    if df.empty:
        raise ValueError(
            "Control CSV contains no rows with a valid gold_relation."
        )

    df["gold_evidence_span"] = df.get(
        "gold_evidence_span",
        ""
    ).fillna("")

    df["event1"] = df.get("event1", "")
    df["event2"] = df.get("event2", "")
    return df


def export_predictions(model, loader, df, device, model_name, threshold, out_path, enforce_gate):
    model.eval()
    gold_map = df.set_index("note_id")["gold_relation"].to_dict()
    rows = []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            outputs = model.predict(ids, mask, threshold, enforce_gate=enforce_gate)
            for example_id, out in zip(batch["note_ids"], outputs):
                answer_idx = out["answer"]
                answer = INV_LABELS[answer_idx] if answer_idx is not None else None
                gold = gold_map[example_id]
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


def make_loader(df, tokenizer, args, pad_labels):
    dataset = RelationDataset(
        df,
        tokenizer,
        args.max_len,
        include_evidence_labels=pad_labels,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", default="bert_control_predictions")
    p.add_argument("--eval_csv", default=None,
                   help="Optional control CSV for inference-only evaluation.")
    p.add_argument("--eval_only", action="store_true",
                   help="Skip training and load --checkpoint instead.")
    p.add_argument("--checkpoint", default=None,
                   help="Checkpoint containing a trained ProvenanceFirstBERT state_dict.")
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

    gated = ProvenanceFirstBERT(
        args.model_name,
        gated=True,
        min_coverage=args.min_coverage,
        min_target=args.min_target,
        finetune=args.finetune,
    ).to(device)

    if args.eval_only:
        if not args.checkpoint:
            raise ValueError("--eval_only requires --checkpoint")
        state = torch.load(args.checkpoint, map_location=device)
        gated.load_state_dict(state)
    else:
        train_df, dev_df, test_df = load_clean_splits(args.data_dir, args.seed)
        train_loader = DataLoader(
            RelationDataset(train_df, tokenizer, args.max_len, True),
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
        )
        steps = train_model(
            gated, train_loader, device, args.epochs, args.lr,
            args.gold_weight, args.floor_weight,
        )
        if steps == 0:
            raise RuntimeError("No optimizer steps were executed.")
        checkpoint = os.path.join(args.out_dir, "bert_provenance_first_state.pt")
        torch.save(gated.state_dict(), checkpoint)
        print(f"saved checkpoint to {checkpoint}")

        for split_name, split_df in [("dev", dev_df), ("test", test_df)]:
            loader = make_loader(split_df, tokenizer, args, True)
            export_predictions(
                gated, loader, split_df, device, "bert_provenance_first",
                args.evidence_threshold,
                os.path.join(args.out_dir, f"bert_provenance_first_{split_name}_predictions.csv"),
                True,
            )

    if args.eval_csv:
        control_df = load_control_csv(args.eval_csv)
        control_loader = make_loader(control_df, tokenizer, args, False)
        name = os.path.splitext(os.path.basename(args.eval_csv))[0]
        export_predictions(
            gated, control_loader, control_df, device,
            "bert_provenance_first",
            args.evidence_threshold,
            os.path.join(args.out_dir, f"bert_provenance_first_{name}_predictions.csv"),
            True,
        )

    with open(os.path.join(args.out_dir, "bert_control_run_metadata.json"), "w") as f:
        json.dump({"config": vars(args), "device": str(device)}, f, indent=2)


if __name__ == "__main__":
    main()
