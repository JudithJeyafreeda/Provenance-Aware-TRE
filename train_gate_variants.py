#!/usr/bin/env python3
"""


Training conditions (all share the same clean training notes, availability = 1):

  A  markers      + remove_evidence and mask_temporal_cue controls (availability 0)
                  -> the recipe of the paper; should reproduce "never abstains
                     on silent removal"
  B  markerfree   + silent_removal controls (availability 0)
                  + silent_keep rewrites   (availability 1, the matched negative
                    control, so "was rewritten" cannot predict unavailability)
  C  mixed        A + B
  D  removalonly  + silent_removal controls (availability 0) WITHOUT silent_keep
                  -> shows whether the gate then learns "rewritten text" instead
                     of "missing evidence" (check its silent_keep abstention)
  E  decorrelated C, but the notice-appended controls (intact note + omission
                  notice) are POSITIVES (availability 1), because their evidence
                  is unchanged. remove_evidence and mask-token controls stay
                  negatives. Neither the marker nor the editing alone then
                  predicts the label.


"""
import argparse
import json
import os
import random

import numpy as np
import pandas as pd

RELATIONS = ["EVENT1-BEFORE-EVENT2", "EVENT2-BEFORE-EVENT1", "EVENT1-OVERLAP-EVENT2"]
REL2ID = {r: i for i, r in enumerate(RELATIONS)}
PREFIX = "gate"


# --------------------------------------------------------------------------- data
def load_data(D, test_ids_csv):
    notes = pd.read_csv(os.path.join(D, "synthetic_notes_v1.csv"))
    adm = notes[notes["is_admission"].astype(str).str.lower() == "true"].copy()
    adm["base"] = adm["note_id"].astype(str)
    adm["pid"] = adm["patient_id"].astype(str)

    marker = pd.read_csv(os.path.join(D, "evidence_removed_only.csv"))
    marker["base"] = marker["base_note_id"].astype(str)
    marker["eid"] = marker["note_id"].astype(str)
    # notice-appended = mask_temporal_cue control where no cue phrase was found
    # (masked_cue False): the note is intact and a notice sentence is appended
    tok = marker["masked_cue"].map(lambda x: str(x).strip().lower() in ("true", "1", "1.0"))
    marker["is_notice"] = (marker["variant"] == "mask_temporal_cue") & ~tok

    sc = os.path.join(D, "silent_controls")
    rem = pd.read_csv(os.path.join(sc, "silent_removal_controls.csv"))
    keep = pd.read_csv(os.path.join(sc, "silent_keep_controls.csv"))
    for df in (rem, keep):
        df["base"] = df["base_note_id"].astype(str)
        df["eid"] = df["note_id"].astype(str)

    qpath = os.path.join(sc, "silent_removal_controls_qwen.csv")
    if os.path.exists(qpath):
        q = pd.read_csv(qpath)[["note_id", "validity_tier"]].rename(
            columns={"note_id": "eid", "validity_tier": "tier_qwen"})
        rem = rem.merge(q, on="eid", how="left")

    test_ids = set(pd.read_csv(test_ids_csv)["example_id"].astype(str))
    test_pids = set(adm.loc[adm["base"].isin(test_ids), "pid"])
    if len(test_pids) != len(test_ids):
        raise SystemExit(f"test ids map to {len(test_pids)} patients, expected {len(test_ids)}")
    return adm, marker, rem, keep, test_pids


def split(adm, test_pids, n_dev, seed):
    rest = sorted(set(adm["pid"]) - test_pids)
    rng = random.Random(seed)
    rng.shuffle(rest)
    dev = set(rest[:n_dev])
    train = set(rest[n_dev:])
    return train, dev


def rows(df, text_col, avail, label_from, kind):
    out = pd.DataFrame({
        "example_id": df["eid"] if "eid" in df else df["base"],
        "base": df["base"],
        "text": df[text_col].astype(str),
        "avail": avail,
        "kind": kind,
    })
    out["gold"] = out["base"].map(label_from)
    return out


def build_train(cond, adm, marker, rem, keep, train_pids, neg_tiers):
    gold = dict(zip(adm["base"], adm["gold_relation"]))
    tb = set(adm.loc[adm["pid"].isin(train_pids), "base"])
    parts = [rows(adm[adm["base"].isin(tb)], "text", 1, gold, "clean")]
    if cond in ("A", "C"):
        parts.append(rows(marker[marker["base"].isin(tb)], "text", 0, gold, "marker"))
    if cond == "E":
        mk = marker[marker["base"].isin(tb)]
        parts.append(rows(mk[~mk["is_notice"]], "text", 0, gold, "marker"))
        parts.append(rows(mk[mk["is_notice"]], "text", 1, gold, "notice_intact"))
    if cond in ("B", "C", "D", "E"):
        r = rem[rem["base"].isin(tb) & rem["validity_tier"].isin(neg_tiers)]
        parts.append(rows(r, "text", 0, gold, "silent_removal"))
    if cond in ("B", "C", "E"):
        parts.append(rows(keep[keep["base"].isin(tb)], "text", 1, gold, "silent_keep"))
    return pd.concat(parts, ignore_index=True)


def build_test(adm, marker, rem, keep, test_pids):
    gold = dict(zip(adm["base"], adm["gold_relation"]))
    tb = set(adm.loc[adm["pid"].isin(test_pids), "base"])
    sets = {
        "clean": rows(adm[adm["base"].isin(tb)], "text", 1, gold, "clean"),
        "unavailable": rows(marker[marker["base"].isin(tb)], "text", 0, gold, "marker"),
        "silent_removal_controls": rows(rem[rem["base"].isin(tb)], "text", 0, gold, "silent_removal"),
        "silent_keep_controls": rows(keep[keep["base"].isin(tb)], "text", 1, gold, "silent_keep"),
    }
    return sets


# --------------------------------------------------------------------------- model
def train_and_predict(train_df, test_sets, args, seed):
    """Returns {set_name: DataFrame with availability_probability, relation probs}."""
    if args.fake_model:   # pipeline test without torch: random scores
        rng = np.random.default_rng(seed)
        out = {}
        for k, df in test_sets.items():
            p = rng.random((len(df), 3))
            out[k] = (rng.random(len(df)), p / p.sum(1, keepdims=True))
        return out

    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.encoder)
    enc = AutoModel.from_pretrained(args.encoder).to(dev)
    H = enc.config.hidden_size

    class Gate(nn.Module):
        def __init__(self):
            super().__init__()
            self.drop = nn.Dropout(0.1)
            self.rel = nn.Linear(H, 3)
            self.av = nn.Linear(H, 1)

        def forward(self, pooled):
            x = self.drop(pooled)
            return self.rel(x), self.av(x).squeeze(-1)

    head = Gate().to(dev)

    def pool(texts, grad):
        b = tok(list(texts), truncation=True, max_length=args.max_len,
                padding=True, return_tensors="pt").to(dev)
        with torch.set_grad_enabled(grad):
            h = enc(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1).clamp(min=1.0)

    texts = train_df["text"].tolist()
    y_rel = torch.tensor([REL2ID[g] for g in train_df["gold"]], dtype=torch.long)
    y_av = torch.tensor(train_df["avail"].values, dtype=torch.float)
    rel_mask = torch.ones(len(train_df))
    if not args.rel_loss_on_negatives:
        rel_mask = torch.tensor((train_df["avail"].values == 1).astype(float), dtype=torch.float)

    if args.unfreeze:
        enc.train()
        params = [{"params": enc.parameters(), "lr": args.lr},
                  {"params": head.parameters(), "lr": args.head_lr}]
        cache = None
    else:
        enc.eval()
        for p in enc.parameters():
            p.requires_grad_(False)
        params = [{"params": head.parameters(), "lr": args.lr}]
        # frozen encoder: pool once, train the heads on cached features
        with torch.no_grad():
            cache = torch.cat([pool(texts[i:i + 64], False).cpu()
                               for i in range(0, len(texts), 64)])
    opt = torch.optim.AdamW(params)
    ce = nn.CrossEntropyLoss(reduction="none")
    bce = nn.BCEWithLogitsLoss()

    idx = np.arange(len(train_df))
    rng = np.random.default_rng(seed)
    for ep in range(args.epochs):
        head.train()
        rng.shuffle(idx)
        tot = 0.0
        for i in range(0, len(idx), args.batch_size):
            j = idx[i:i + args.batch_size]
            if cache is not None:
                pooled = cache[j].to(dev)
            else:
                pooled = pool([texts[k] for k in j], True)
            lr_, la = head(pooled)
            m = rel_mask[j].to(dev)
            l_rel = (ce(lr_, y_rel[j].to(dev)) * m).sum() / m.sum().clamp(min=1.0)
            loss = l_rel + args.avail_weight * bce(la, y_av[j].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(j)
        print(f"    epoch {ep + 1}/{args.epochs}  loss {tot / len(idx):.4f}", flush=True)

    head.eval()
    enc.eval()
    out = {}
    with torch.no_grad():
        for k, df in test_sets.items():
            av, rp = [], []
            t = df["text"].tolist()
            for i in range(0, len(t), 64):
                lr_, la = head(pool(t[i:i + 64], False))
                av.append(torch.sigmoid(la).cpu().numpy())
                rp.append(torch.softmax(lr_, -1).cpu().numpy())
            out[k] = (np.concatenate(av) if av else np.array([]),
                      np.concatenate(rp) if rp else np.zeros((0, 3)))
    del enc, head
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return out


def write_preds(df, av, rp, path, thr):
    arg = [RELATIONS[i] for i in rp.argmax(1)] if len(rp) else []
    ab = av < thr
    ans = [None if a else r for a, r in zip(ab, arg)]
    pd.DataFrame({
        "example_id": df["example_id"].values,
        "answer": ans,
        "abstained": ab,
        "correct": [None if a else float(r == g) for a, r, g in zip(ab, arg, df["gold"])],
        "availability_probability": av,
        "argmax_relation": arg,
        "max_relation_prob": rp.max(1) if len(rp) else [],
    }).to_csv(path, index=False)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--test_ids_csv", required=True,
                    help="original clean test predictions (defines the 150 test patients)")
    ap.add_argument("--out_dir", default="output/gate_variants")
    ap.add_argument("--conditions", default="A,B,C,D")
    ap.add_argument("--seeds", default="7,13,21,42,101")
    ap.add_argument("--encoder", default="emilyalsentzer/Bio_ClinicalBERT")
    ap.add_argument("--unfreeze", action="store_true")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=5e-5, help="frozen: head lr; unfrozen: encoder lr")
    ap.add_argument("--head_lr", type=float, default=1e-3, help="head lr when --unfreeze")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--avail_weight", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--n_dev", type=int, default=150)
    ap.add_argument("--split_seed", type=int, default=7)
    ap.add_argument("--neg_tiers", default="strict")
    ap.add_argument("--no_rel_loss_on_negatives", dest="rel_loss_on_negatives", action="store_false")
    ap.add_argument("--dry_run", action="store_true", help="build and count the data sets, no training")
    ap.add_argument("--fake_model", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    conds = [c.strip().upper() for c in args.conditions.split(",") if c.strip()]
    seeds = [int(s) for s in args.seeds.split(",")]
    tiers = [t.strip() for t in args.neg_tiers.split(",")]
    enc_tag = "unfrozen" if args.unfreeze else "frozen"

    adm, marker, rem, keep, test_pids = load_data(args.data_dir, args.test_ids_csv)
    train_pids, dev_pids = split(adm, test_pids, args.n_dev, args.split_seed)
    tests = build_test(adm, marker, rem, keep, test_pids)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"patients: train {len(train_pids)}, dev {len(dev_pids)} (unused), test {len(test_pids)}")
    print("test sets: " + ", ".join(f"{k} {len(v)}" for k, v in tests.items()))

    manifest = []
    for c in conds:
        tr = build_train(c, adm, marker, rem, keep, train_pids, tiers)
        comp = tr.groupby(["kind", "avail"]).size().to_dict()
        print(f"condition {c}: {len(tr)} training examples  " +
              ", ".join(f"{k}/avail={a}: {n}" for (k, a), n in sorted(comp.items())))
        for (k, a), n in comp.items():
            manifest.append({"condition": c, "kind": k, "avail": a, "n": n,
                             "neg_tiers": args.neg_tiers, "encoder": enc_tag})
        if args.dry_run:
            continue
        for s in seeds:
            run = os.path.join(args.out_dir, f"{c}_{enc_tag}_seed{s}")
            if os.path.exists(os.path.join(run, f"{PREFIX}_silent_keep_controls_test_predictions.csv")):
                print(f"  {run} exists, skipping")
                continue
            os.makedirs(run, exist_ok=True)
            print(f"  training {c} / {enc_tag} / seed {s}", flush=True)
            preds = train_and_predict(tr, tests, args, s)
            for k, df in tests.items():
                av, rp = preds[k]
                write_preds(df, av, rp, os.path.join(run, f"{PREFIX}_{k}_test_predictions.csv"), args.threshold)
            with open(os.path.join(run, "config.json"), "w") as f:
                json.dump({**vars(args), "condition": c, "seed": s}, f, indent=2)
    pd.DataFrame(manifest).to_csv(os.path.join(args.out_dir, f"manifest_{enc_tag}.csv"), index=False)
    print(f"done -> {args.out_dir}")


if __name__ == "__main__":
    main()
