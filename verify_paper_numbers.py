#!/usr/bin/env python3
"""Recompute every data-derived number in the manuscript from the saved outputs and
compare it with the value printed in the paper.

Every row reports: section, quantity, paper value, recomputed value, status.
Status is OK when the recomputed value rounds to the printed value, MISMATCH otherwise,
and MISSING when the input file for that number was not found. Sections whose
inputs are absent are skipped, not failed.
"""
import argparse
import glob
import json
import math
import os
import random
import re

import numpy as np
import pandas as pd

E12, E21, OVL = "EVENT1-BEFORE-EVENT2", "EVENT2-BEFORE-EVENT1", "EVENT1-OVERLAP-EVENT2"
CLASSES = [E12, OVL, E21]
CUES = ["prior to", "before", "preceding", "following", "after",
        "subsequent to", "during", "concurrent with", "while undergoing"]
SUFFIXES = ("_remove_evidence", "_mask_temporal_cue", "_silent_removal", "_silent_keep")
G = "provenance_first_availability_supervised"
R = "regular_bert"

ROWS = []


def check(section, name, paper, value, digits=3, pct=False):
    """paper: number as printed (fraction or percent if pct=True); value: fraction."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        ROWS.append((section, name, paper, None, "MISSING"))
        return
    shown = value * 100 if pct else value
    from decimal import Decimal, ROUND_HALF_UP
    q = Decimal(1).scaleb(-digits)
    ok = Decimal(repr(float(shown))).quantize(q, ROUND_HALF_UP) == Decimal(repr(float(paper))).quantize(q, ROUND_HALF_UP)
    ROWS.append((section, name, paper, round(float(shown), digits + 2), "OK" if ok else "MISMATCH"))


def count(section, name, paper, value):
    ROWS.append((section, name, paper, value, "OK" if value == paper else "MISMATCH"))


def as_bool(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "1.0"}


def base_id(x):
    for s in SUFFIXES:
        if x.endswith(s):
            return x[: -len(s)]
    return x


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    ranks = pd.Series(np.concatenate([pos, neg])).rank(method="average").to_numpy()
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def threshold_for_coverage(scores, target):
    s = np.asarray(scores, float)
    best, gap = np.inf, np.inf
    for t in np.unique(s[np.isfinite(s)]):
        g = abs(np.mean(s >= t) - target)
        if g < gap - 1e-12:
            best, gap = t, g
    return best


def load_preds(path, gold):
    """Prediction CSV -> example_id, base, gold, answer, abstained, correct."""
    if not path or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["example_id"] = df["example_id"].astype(str)
    df["base"] = df["example_id"].map(base_id)
    df["gold"] = df["base"].map(gold)
    if df["gold"].isna().any():
        raise ValueError(f"{path}: {int(df['gold'].isna().sum())} rows without a gold relation")
    df["abst"] = df["abstained"].map(as_bool)
    df["answer"] = df["answer"].where(~df["abst"], None)
    df["ok"] = (~df["abst"]) & (df["answer"].astype(str) == df["gold"])
    return df


def summ(df):
    n = len(df)
    a = int((~df.abst).sum())
    c = int(df.ok.sum())
    return n, a, c


def find(d, pattern):
    hits = sorted(glob.glob(os.path.join(d, pattern))) if d else []
    return hits[0] if hits else None


def paired(ctrl, clean):
    c = clean[clean.example_id.isin(set(ctrl.base))]
    return summ(c)


# ---------------------------------------------------------------- sections
def corpus(D, out):
    notes = pd.read_csv(os.path.join(D, "synthetic_notes_v1.csv"))
    rel = notes[notes.gold_relation.notna() & notes.gold_evidence_span.notna()].reset_index(drop=True)
    s = "3.1 corpus"
    count(s, "relation notes", 1000, len(rel))
    vc = rel.gold_relation.value_counts()
    count(s, "EVENT1-BEFORE-EVENT2", 345, int(vc.get(E12, 0)))
    count(s, "EVENT2-BEFORE-EVENT1", 301, int(vc.get(E21, 0)))
    count(s, "EVENT1-OVERLAP-EVENT2", 354, int(vc.get(OVL, 0)))
    check(s, "majority-class rate", 0.354, vc.max() / len(rel))
    count(s, "gold spans that are substrings of the note (paper: none)", 0,
          int(sum(sp in t for sp, t in zip(rel.gold_evidence_span, rel.text))))
    count(s, "median words per note", 33, int(rel.text.str.split().str.len().median()))
    pats = list(rel.patient_id.unique())
    random.Random(7).shuffle(pats)
    n = int(0.15 * len(pats))
    test = set(pats[:n])
    dev = set(pats[n:2 * n])
    count(s, "test patients", 150, len(test))
    count(s, "dev patients", 150, len(dev))
    count(s, "train patients", 700, len(pats) - 2 * n)

    def cue(t):
        t = t.lower()
        return next((c for c in CUES if c in t), None)
    rel["cue"] = rel.text.map(cue)
    rel["test"] = rel.patient_id.isin(test)
    count("3.4 marker controls", "notice appended, all notes", 579, int(rel.cue.isna().sum()))
    count("3.4 marker controls", "notice appended, test notes", 89, int((rel.cue.isna() & rel.test).sum()))
    count("3.4 marker controls", "token inserted, test notes", 61, int((rel.cue.notna() & rel.test).sum()))
    out["rel"] = rel
    out["gold"] = dict(zip(rel.note_id.astype(str), rel.gold_relation))
    out["test_patients"] = test


def marker_table(D, bert_dir, llm_dir, out):
    ctrl_path = os.path.join(D, "evidence_removed_only.csv")
    if not os.path.exists(ctrl_path):
        ROWS.append(("Table 1", "evidence_removed_only.csv", None, None, "MISSING"))
        return
    ctrl = pd.read_csv(ctrl_path)
    ctrl["example_id"] = ctrl["note_id"].astype(str)
    masked = dict(zip(ctrl.example_id, ctrl.masked_cue.map(as_bool)))
    gold = out["gold"]

    def strata(pr):
        pr = pr.copy()
        pr["variant"] = np.where(pr.example_id.str.endswith("_remove_evidence"), "remove", "mask")
        pr["tok"] = pr.example_id.map(masked).fillna(False)
        return {"remove": pr[pr.variant == "remove"],
                "token": pr[(pr.variant == "mask") & pr.tok],
                "notice": pr[(pr.variant == "mask") & ~pr.tok]}

    paper = {  # n, clean cov, answer rate, answered acc, unsafe, clean acc (text)
        "BERT": {"clean": (150, None, 0.987, 0.899, None),
                 "remove": (150, 0.987, 0.000, None, 0.000),
                 "token": (61, 0.967, 0.000, None, 0.000, 0.983),
                 "notice": (89, 1.000, 0.034, 0.667, 0.011, 0.843)},
        "LLM": {"clean": (1000, None, 0.700, 0.663, None),
                "remove": (1000, 0.700, 0.000, None, 0.000),
                "token": (421, 0.853, 0.786, 0.483, 0.406, 0.780),
                "notice": (579, 0.589, 0.000, None, 0.000)},
    }
    files = {
        "BERT": (find(bert_dir, f"{G}_clean_test_predictions.csv"),
                 find(bert_dir, f"{G}_unavailable_test_predictions.csv")),
        "LLM": (find(llm_dir, "llm_availability_gated_synthetic_notes_relation_only_predictions.csv"),
                find(llm_dir, "llm_availability_gated_evidence_removed_only_predictions.csv")),
    }
    tot = {}
    for gate, (cp, up) in files.items():
        clean, unav = load_preds(cp, gold), load_preds(up, gold)
        s = f"Table 1 {gate}"
        if clean is None or unav is None:
            ROWS.append((s, "prediction files", None, None, "MISSING"))
            continue
        out[f"{gate}_clean"] = clean
        n, a, c = summ(clean)
        p = paper[gate]["clean"]
        count(s, "clean n", p[0], n)
        check(s, "clean answer rate", p[2], a / n)
        check(s, "clean answered acc", p[3], c / a)
        if gate == "LLM":
            lo, hi = wilson(a, n)
            check(s, "clean cov Wilson low", 0.671, lo)
            check(s, "clean cov Wilson high", 0.728, hi)
        st = strata(unav)
        abst_total = 0
        for k, sub in st.items():
            pp = paper[gate][k]
            n, a, c = summ(sub)
            cn, ca, cc = paired(sub, clean)
            abst_total += n - a
            count(s, f"{k}: n", pp[0], n)
            check(s, f"{k}: clean coverage, same notes", pp[1], ca / cn if cn else float("nan"))
            check(s, f"{k}: answer rate", pp[2], a / n)
            if pp[3] is not None:
                check(s, f"{k}: answered acc", pp[3], c / a if a else float("nan"))
            check(s, f"{k}: unsafe-answer rate", pp[4], (a - c) / n)
            if len(pp) > 5:
                check(s, f"{k}: clean answered acc, same notes", pp[5], cc / ca if ca else float("nan"))
            tot[(gate, k)] = n - a
        tot[gate] = (abst_total, len(unav))
    if "BERT" in tot:
        count("4.2 text", "BERT abstentions on marker controls (297 of 300)", 297, tot["BERT"][0])
    if "LLM" in tot:
        count("4.2 text", "LLM abstentions on marker controls (1,669 of 2,000)", 1669, tot["LLM"][0])
        verbal = tot[("LLM", "remove")] + tot[("LLM", "notice")]
        count("4.2 text", "LLM abstentions on the two worded inputs (1,579)", 1579, verbal)
        check("4.2 text", "share of LLM abstentions from worded inputs (94.6%)", 94.6,
              verbal / tot["LLM"][0], digits=1, pct=True)


def llm_clean_per_class(out):
    clean = out.get("LLM_clean")
    if clean is None:
        return
    s = "4.1 LLM per class"
    ans = clean[~clean.abst]
    count(s, "answered clean notes (Table 1 says 700)", 700, len(ans))
    check(s, "share of answers that are EVENT1-BEFORE-EVENT2 (63.6%)", 63.6,
          (ans.answer == E12).mean(), digits=1, pct=True)
    count(s, "overlap notes answered (258)", 258, int((ans.gold == OVL).sum()))
    count(s, "EVENT2-BEFORE-EVENT1 notes answered (182)", 182, int((ans.gold == E21).sum()))
    count(s, "overlap notes answered EVENT1-BEFORE-EVENT2 (141)", 141,
          int(((ans.gold == OVL) & (ans.answer == E12)).sum()))
    count(s, "E2<E1 notes answered EVENT1-BEFORE-EVENT2 (49)", 49,
          int(((ans.gold == E21) & (ans.answer == E12)).sum()))
    for cls, acc, cov in [(E12, 0.981, 0.754), (OVL, 0.453, 0.729), (E21, 0.505, 0.605)]:
        sub = clean[clean.gold == cls]
        a = sub[~sub.abst]
        check(s, f"{cls} answered acc", acc, a.ok.mean())
        check(s, f"{cls} clean coverage", cov, len(a) / len(sub))


def silent_tables(D, bert_dir, llm_dir, out):
    sc = os.path.join(D, "silent_controls")
    rem_p, keep_p = os.path.join(sc, "silent_removal_controls.csv"), os.path.join(sc, "silent_keep_controls.csv")
    if not (os.path.exists(rem_p) and os.path.exists(keep_p)):
        ROWS.append(("Table 2", "silent control CSVs", None, None, "MISSING"))
        return
    rem, keep = pd.read_csv(rem_p), pd.read_csv(keep_p)
    tier = dict(zip(rem.note_id.astype(str), rem.validity_tier))
    s = "3.5 silent controls"
    count(s, "silent_removal accepted", 507, len(rem))
    count(s, "strict", 238, int((rem.validity_tier == "strict").sum()))
    count(s, "implied-only", 269, int((rem.validity_tier == "implied_only").sum()))
    count(s, "silent_keep accepted", 452, len(keep))
    tp = out["test_patients"]
    count(s, "BERT test silent_removal", 81, int(rem.patient_id.isin(tp).sum()))
    count(s, "BERT test strict", 37, int((rem.patient_id.isin(tp) & (rem.validity_tier == "strict")).sum()))
    count(s, "BERT test silent_keep", 70, int(keep.patient_id.isin(tp).sum()))
    rep = os.path.join(sc, "silent_controls_report.json")
    if os.path.exists(rep):
        r = json.load(open(rep))
        v = next((r[k] for k in ("n_orig_valid", "orig_valid", "n_original_valid") if k in r), None)
        if v is not None:
            count(s, "notes whose original the judge recovered", 561, v)

    gold = out["gold"]
    paper = {  # n, clean cov, clean acc, edited cov, edited acc
        "BERT": {"keep": (70, .986, .928, .957, .836), "implied_only": (44, .977, .930, 1.0, .614),
                 "strict": (37, 1.0, .946, 1.0, .568), "all": (81, .988, .938, 1.0, .593)},
        "LLM": {"keep": (452, .723, .881, .721, .850), "implied_only": (269, .651, .851, .688, .384),
                "strict": (238, .756, .839, .277, .515), "all": (507, .700, .845, .495, .418)},
    }
    files = {
        "BERT": (find(bert_dir, f"{G}_silent_removal_controls*_predictions.csv"),
                 find(bert_dir, f"{G}_silent_keep_controls*_predictions.csv")),
        "LLM": (find(llm_dir, "llm_availability_gated_silent_removal_controls_predictions.csv"),
                find(llm_dir, "llm_availability_gated_silent_keep_controls_predictions.csv")),
    }
    for gate, (rp, kp) in files.items():
        clean = out.get(f"{gate}_clean")
        r, k = load_preds(rp, gold), load_preds(kp, gold)
        st = f"Table 2 {gate}"
        if clean is None or r is None or k is None:
            ROWS.append((st, "prediction files", None, None, "MISSING"))
            continue
        r["tier"] = r.example_id.map(tier)
        out[f"{gate}_rem"], out[f"{gate}_keep"] = r, k
        groups = {"keep": k, "implied_only": r[r.tier == "implied_only"],
                  "strict": r[r.tier == "strict"], "all": r}
        for name, sub in groups.items():
            pp = paper[gate][name]
            n, a, c = summ(sub)
            cn, ca, cc = paired(sub, clean)
            count(st, f"{name}: n", pp[0], n)
            check(st, f"{name}: clean coverage", pp[1], ca / cn if cn else float("nan"))
            check(st, f"{name}: clean acc", pp[2], cc / ca if ca else float("nan"))
            check(st, f"{name}: edited coverage", pp[3], a / n)
            check(st, f"{name}: edited acc", pp[4], c / a if a else float("nan"))


def per_class(out):
    r, k, clean = out.get("BERT_rem"), out.get("BERT_keep"), out.get("BERT_clean")
    if r is not None and clean is not None:
        s = "Table 3 BERT"
        paper = {E12: (28, 1.000, .750, (21, 0, 7), 26, 1.000, .960),
                 OVL: (23, .913, .609, (9, 0, 14), 19, .842, .882),
                 E21: (30, .900, .433, (4, 13, 13), 25, .920, .680)}
        for cls, pp in paper.items():
            sub = r[r.gold == cls]
            n, a, c = summ(sub)
            cn, ca, cc = paired(sub, clean)
            count(s, f"{cls} removal n", pp[0], n)
            check(s, f"{cls} clean acc", pp[1], cc / ca if ca else float("nan"))
            check(s, f"{cls} edited acc", pp[2], c / a if a else float("nan"))
            got = tuple(int((sub.answer == x).sum()) for x in (E12, E21, OVL))
            count(s, f"{cls} answered as E1<E2/E2<E1/Ovl", pp[3], got)
            ks = k[k.gold == cls]
            n, a, c = summ(ks)
            cn, ca, cc = paired(ks, clean)
            count(s, f"{cls} keep n", pp[4], n)
            check(s, f"{cls} keep clean acc", pp[5], cc / ca if ca else float("nan"))
            check(s, f"{cls} keep edited acc", pp[6], c / a if a else float("nan"))
    r, k, clean = out.get("LLM_rem"), out.get("LLM_keep"), out.get("LLM_clean")
    if r is not None:
        s = "4.4 LLM per class"
        ans = r[~r.abst]
        e21 = ans[ans.gold == E21]
        count(s, "E2<E1 removals answered (70)", 70, len(e21))
        count(s, "E2<E1 removals answered E2<E1 (0)", 0, int((e21.answer == E21).sum()))
        check(s, "E2<E1 removals answered overlap (66%)", 66, (e21.answer == OVL).mean(), digits=0, pct=True)
        e12 = ans[ans.gold == E12]
        check(s, "E1<E2 removals answered overlap (55%)", 55, (e12.answer == OVL).mean(), digits=0, pct=True)
        ov = ans[ans.gold == OVL]
        check(s, "overlap removal acc (0.747)", 0.747, ov.ok.mean())
        if clean is not None:
            cl = clean[clean.example_id.isin(set(r.base)) & (clean.gold == OVL) & ~clean.abst]
            check(s, "overlap clean acc, same notes (0.744)", 0.744, cl.ok.mean())
        strict = ans[ans.tier == "strict"]
        so = strict[strict.gold == OVL]
        count(s, "strict overlap answered (27)", 27, len(so))
        count(s, "strict overlap correct (25)", 25, int(so.ok.sum()))
        se = strict[strict.gold == E21]
        count(s, "strict E2<E1 answered (15)", 15, len(se))
        count(s, "strict E2<E1 correct (0)", 0, int(se.ok.sum()))
        if k is not None:
            ka = k[~k.abst]
            for cls, v in [(E12, 1.000), (OVL, 0.768), (E21, 0.705)]:
                check(s, f"silent_keep acc {cls}", v, ka[ka.gold == cls].ok.mean())


def baselines(D, bert_dir, llm_dir, rerun_dirs, out):
    gold = out["gold"]
    rem = out.get("BERT_rem")
    tier = None
    sc = os.path.join(D, "silent_controls", "silent_removal_controls.csv")
    if os.path.exists(sc):
        t = pd.read_csv(sc)
        tier = dict(zip(t.note_id.astype(str), t.validity_tier))

    def scored(path, pred_col, score_col):
        if not path or not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        df["example_id"] = df["example_id"].astype(str)
        if pred_col not in df.columns or score_col not in df.columns:
            print(f"[baselines] {path} lacks {sorted({pred_col, score_col} - set(df.columns))}")
            return None
        pred = df[pred_col]
        has = pred.notna() & pred.astype(str).ne("None") & pred.astype(str).ne("")
        sc_ = pd.to_numeric(df[score_col], errors="coerce")
        df["_s"] = np.where(has & sc_.notna(), sc_, -1e9)
        df["tier"] = df.example_id.map(tier) if tier else None
        return df

    def block(sec, label, files, pred_col, score_col, target, paper):
        clean = scored(files["clean"], pred_col, score_col)
        if clean is None:
            print(f"[baselines] {label}: clean file = {files['clean']}")
            ROWS.append((sec, f"{label}: files/columns", None, None, "MISSING"))
            return
        t = threshold_for_coverage(clean._s, target)
        conds = {"marker": scored(files["marker"], pred_col, score_col),
                 "removal_all": scored(files["removal"], pred_col, score_col),
                 "keep": scored(files["keep"], pred_col, score_col)}
        if conds["removal_all"] is not None:
            conds["strict"] = conds["removal_all"][conds["removal_all"].tier == "strict"]
        if "clean_cov" in paper:
            check(sec, f"{label}: achieved clean coverage", paper["clean_cov"], float(np.mean(clean._s >= t)))
        for name, d in conds.items():
            if d is None:
                continue
            for metric, val in [("abst", 1 - float(np.mean(d._s >= t))), ("auroc", auroc(clean._s, d._s))]:
                key = f"{name}_{metric}"
                if key in paper:
                    check(sec, f"{label}: {name} {metric}", paper[key], val)

    bfiles = lambda prefix: {
        "clean": find(bert_dir, f"{prefix}_clean_test_predictions.csv"),
        "marker": find(bert_dir, f"{prefix}_unavailable_test_predictions.csv"),
        "removal": find(bert_dir, f"{prefix}_silent_removal_controls*_predictions.csv"),
        "keep": find(bert_dir, f"{prefix}_silent_keep_controls*_predictions.csv")}
    s = "Table 4 baselines"
    # the gate's answer column is empty when it abstains, so score the argmax for baselines
    block(s, "BERT availability score", bfiles(G), "argmax_relation",
          "availability_probability", 148 / 150,
          {"marker_abst": .990, "marker_auroc": .998, "strict_abst": 0.0, "strict_auroc": .521,
           "keep_abst": .043, "keep_auroc": .636, "removal_all_auroc": .522})
    block(s, "BERT max softmax, availability model", bfiles(G), "argmax_relation", "max_relation_prob", 148 / 150,
          {"marker_abst": .030, "marker_auroc": .765, "strict_abst": 0.0, "strict_auroc": .662,
           "keep_abst": .129, "keep_auroc": .665, "removal_all_auroc": .642})
    block(s, "BERT max softmax, no availability loss", bfiles(R), "argmax_relation", "max_relation_prob", 148 / 150,
          {"marker_abst": .010, "marker_auroc": .735, "strict_abst": 0.0, "strict_auroc": .741,
           "keep_abst": .043, "keep_auroc": .693, "removal_all_auroc": .761, "removal_all_abst": .012})
    src = rerun_dirs[0] if rerun_dirs else llm_dir
    lf = {"clean": find(src, "llm_availability_gated_synthetic_notes_relation_only_predictions.csv"),
          "marker": find(src, "llm_availability_gated_evidence_removed_only_predictions.csv"),
          "removal": find(src, "llm_availability_gated_silent_removal_controls_predictions.csv"),
          "keep": find(src, "llm_availability_gated_silent_keep_controls_predictions.csv")}
    block(s, "LLM verbalised confidence", lf, "raw_relation", "llm_confidence", 0.700,
          {"clean_cov": .602, "marker_abst": .836, "marker_auroc": .873, "strict_abst": .996,
           "strict_auroc": .876, "keep_abst": .394, "keep_auroc": .498, "removal_all_auroc": .820,
           "removal_all_abst": 1 - .037})
    block(s, "LLM re-query agreement", lf, "raw_relation", "reask_agreement", 0.700,
          {"clean_cov": .730, "marker_abst": .826, "marker_auroc": .881, "strict_abst": .912,
           "strict_auroc": .886, "keep_abst": .235, "keep_auroc": .481, "removal_all_auroc": .770})


def reruns(llm_dir, rerun_dirs, out):
    if not rerun_dirs:
        ROWS.append(("Reproducibility", "--llm_rerun_dirs", None, None, "MISSING"))
        return
    gold = out["gold"]
    stems = {"clean": "synthetic_notes_relation_only", "marker": "evidence_removed_only",
             "removal": "silent_removal_controls", "keep": "silent_keep_controls"}
    paper_cov = {"clean": [0.700, 0.699, 0.694], "marker": [0.166, 0.164],
                 "removal": [0.495, 0.495], "keep": [0.721, 0.723]}
    runs = [llm_dir] + list(rerun_dirs)
    s = "Reproducibility"
    preds = {}
    for i, d in enumerate(runs):
        for key, stem in stems.items():
            p = load_preds(find(d, f"llm_availability_gated_{stem}_predictions.csv"), gold)
            if p is None:
                ROWS.append((s, f"run {i} {key}: {d}/llm_availability_gated_{stem}_predictions.csv", None, None, "MISSING"))
                continue
            preds[(i, key)] = p
            if i < len(paper_cov[key]):
                check(s, f"run {i} {key} coverage", paper_cov[key][i], float((~p.abst).mean()))
    if (0, "clean") in preds and (1, "clean") in preds:
        a = preds[(0, "clean")].set_index("example_id").abst
        b = preds[(1, "clean")].set_index("example_id").abst.reindex(a.index)
        check(s, "run 0 vs run 1: identical clean abstain/answer decisions (98.7%)", 98.7,
              float((a == b).mean()), digits=1, pct=True)
    if (1, "clean") in preds and (2, "clean") in preds:
        a = preds[(1, "clean")].set_index("example_id").abst
        b = preds[(2, "clean")].set_index("example_id").abst.reindex(a.index)
        ROWS.append((s, "run 1 vs run 2: identical clean decisions (not in paper)", None,
                     round(float((a == b).mean()), 4), "INFO"))


def spans(D, llm_dir, out, recall_csv=None):
    gp = os.path.join(D, "gold_spans_v2", "gold_spans_v2.csv")
    s = "4.6 cited evidence"
    if not os.path.exists(gp):
        ROWS.append((s, "gold_spans_v2.csv", None, None, "MISSING"))
        return
    g = pd.read_csv(gp)
    vc = g.reason.value_counts()
    count(s, "accepted spans", 482, int(vc.get("accepted", 0)))
    count(s, "lost in paraphrase", 139, int(vc.get("evidence_lost_in_paraphrase", 0)))
    count(s, "attempts exhausted", 379, int(vc.get("attempts_exhausted", 0)))
    g["gold"] = g.note_id.astype(str).map(out["gold"])
    acc = g.reason == "accepted"
    for cls, v in [(OVL, 71), (E12, 41), (E21, 29)]:
        check(s, f"span recovered, {cls} (%)", v, acc[g.gold == cls].mean(), digits=0, pct=True)
    words = g.text.str.split().str.len()
    count(s, "median words, notes with span", 33, int(words[acc].median()))
    count(s, "median words, notes without span", 32, int(words[~acc].median()))
    rp = recall_csv or find(llm_dir, "llm_evidence_recall_v2.csv")
    if not rp:
        ROWS.append((s, "llm_evidence_recall_v2.csv", None, None, "MISSING"))
        return
    m = pd.read_csv(rp)
    m["abst"] = m.abstained.map(as_bool)
    m = m[m.evidence_recall_v2.notna()]
    count(s, "scorable notes, all (abstained included)", 482, len(m))
    a = m[~m.abst].copy()
    count(s, "scorable notes, answered (paper: 379)", 379, len(a))
    check(s, "mean recall, answered", 0.850, a.evidence_recall_v2.astype(float).mean())
    check(s, "mean precision, answered", 0.864, a.evidence_precision_v2.astype(float).mean())
    ROWS.append((s, "mean recall, all scorable incl. abstained (not in paper)", None, round(float(m.evidence_recall_v2.astype(float).mean()), 4), "INFO"))
    frac = a.predicted_span_len_frac_of_note.astype(float)
    check(s, "cited span as % of note", 33.9, frac.mean(), digits=1, pct=True)
    count(s, "cited spans covering >80% of note", 0, int((frac > 0.8).sum()))
    a["gold"] = a.example_id.astype(str).map(out["gold"])
    for cls, rv in [(E12, .890), (OVL, .804), (E21, .917)]:
        check(s, f"recall {cls}", rv, a[a.gold == cls].evidence_recall_v2.astype(float).mean())
    check(s, f"precision {E12}", 0.785, a[a.gold == E12].evidence_precision_v2.astype(float).mean())


def independent_judge(D, out, qwen_csv=None):
    """Table 3 and the Methods/Results numbers on the cross-family (Qwen) re-judging."""
    s = "Table 3 independent judge"
    path = qwen_csv or os.path.join(D, "silent_controls", "silent_removal_controls_qwen.csv")
    if not os.path.exists(path):
        ROWS.append((s, path, None, None, "MISSING"))
        return
    q = pd.read_csv(path)
    q["note_id"] = q["note_id"].astype(str)
    count(s, "re-judged edits", 507, len(q))
    count(s, "judge errors", 0, int((q.validity_tier == "judge_error").sum()))
    ok = q[q.validity_tier != "judge_error"]
    agree = int((ok.validity_tier == ok.validity_tier_judge1).sum())
    count(s, "tier agreement (350 of 507)", 350, agree)
    check(s, "tier agreement (%)", 69.0, agree / len(ok), digits=1, pct=True)
    po = agree / len(ok)
    p1 = (ok.validity_tier_judge1 == "strict").mean()
    p2 = (ok.validity_tier == "strict").mean()
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    check(s, "Cohen's kappa (0.40)", 0.40, (po - pe) / (1 - pe), digits=2)
    count(s, "original strict kept strict (219 of 238)", 219,
          int(((q.validity_tier_judge1 == "strict") & (q.validity_tier == "strict")).sum()))
    count(s, "original implied rated strict (138 of 269)", 138,
          int(((q.validity_tier_judge1 == "implied_only") & (q.validity_tier == "strict")).sum()))
    combo = dict(zip(q.note_id, q.validity_tier_judge1 + "|" + q.validity_tier))
    paper = {  # (orig|indep): n, clean cov, edited cov, edited acc
        "strict|strict": (219, .767, .233, .510),
        "implied_only|strict": (138, .761, .565, .410),
        "strict|implied_only": (19, .632, .789, .533),
        "implied_only|implied_only": (131, .534, .817, .364),
    }
    r, clean = out.get("LLM_rem"), out.get("LLM_clean")
    if r is not None and clean is not None:
        r = r.copy()
        r["combo"] = r.example_id.map(combo)
        for k, (n0, cc, ec, ea) in paper.items():
            sub = r[r.combo == k]
            n, a, c = summ(sub)
            cn, ca, cc_ = paired(sub, clean)
            count(s, f"LLM {k}: n", n0, n)
            check(s, f"LLM {k}: clean coverage", cc, ca / cn if cn else float("nan"))
            check(s, f"LLM {k}: edited coverage", ec, a / n if n else float("nan"))
            check(s, f"LLM {k}: edited accuracy", ea, c / a if a else float("nan"))
    b, bclean = out.get("BERT_rem"), out.get("BERT_clean")
    if b is not None and bclean is not None:
        qtier = dict(zip(q.note_id, q.validity_tier))
        bs = b[b.example_id.map(qtier) == "strict"]
        n, a, c = summ(bs)
        cn, ca, cc_ = paired(bs, bclean)
        count(s, "BERT test notes rated strict by independent judge (57)", 57, n)
        count(s, "BERT abstentions on them (0)", 0, n - a)
        check(s, "BERT clean accuracy on them (0.964)", 0.964, cc_ / ca if ca else float("nan"))
        check(s, "BERT edited accuracy on them (0.614)", 0.614, c / a if a else float("nan"))


def rejudge(path):
    s = "3.5 judge re-run"
    if not path or not os.path.exists(path):
        ROWS.append((s, "--rejudge_csv", None, None, "MISSING"))
        return
    d = pd.read_csv(path)
    count(s, "tier assignments reproduced (500 of 507)", 500,
          int((d.validity_tier == d.validity_tier_judge1).sum()))
    count(s, "rows re-judged", 507, len(d))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--bert_dir", default=None)
    p.add_argument("--llm_dir", default=None)
    p.add_argument("--llm_rerun_dirs", default="")
    p.add_argument("--rejudge_csv", default=None)
    p.add_argument("--baseline_bert_dir", default=None,
                   help="dir with v3 exports (argmax_relation, max_relation_prob); defaults to --bert_dir")
    p.add_argument("--baseline_llm_dir", default=None,
                   help="LLM run with raw_relation/llm_confidence/reask_agreement; defaults to first re-run")
    p.add_argument("--recall_csv", default=None, help="llm_evidence_recall_v2.csv")
    p.add_argument("--qwen_csv", default=None, help="silent_removal_controls_qwen.csv (independent judge)")
    p.add_argument("--out_csv", default="paper_number_check.csv")
    a = p.parse_args()
    reruns_ = [x for x in a.llm_rerun_dirs.split(",") if x]
    out = {}
    corpus(a.data_dir, out)
    marker_table(a.data_dir, a.bert_dir, a.llm_dir, out)
    llm_clean_per_class(out)
    silent_tables(a.data_dir, a.bert_dir, a.llm_dir, out)
    per_class(out)
    baselines(a.data_dir, a.baseline_bert_dir or a.bert_dir, a.llm_dir,
              [a.baseline_llm_dir] if a.baseline_llm_dir else reruns_, out)
    reruns(a.llm_dir, reruns_, out)
    spans(a.data_dir, a.llm_dir, out, a.recall_csv)
    rejudge(a.rejudge_csv)
    independent_judge(a.data_dir, out, a.qwen_csv)
    df = pd.DataFrame(ROWS, columns=["section", "quantity", "paper", "recomputed", "status"])
    df.to_csv(a.out_csv, index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 1000, "display.max_colwidth", 70):
        print(df.to_string(index=False))
    print("\n" + df.status.value_counts().to_string())
    print(f"wrote {a.out_csv}")


if __name__ == "__main__":
    main()
