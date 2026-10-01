#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

RELATIONS = ["EVENT1-BEFORE-EVENT2", "EVENT2-BEFORE-EVENT1", "EVENT1-OVERLAP-EVENT2"]
CACHE_VERSION = 1

RELOCATE_SYSTEM = """You align evidence between two versions of the same clinical note for a
controlled experiment. You are given:
  - ORIGINAL_SPAN: a short sentence from the note BEFORE it was paraphrased, stating
    when Event 1 (diagnosis) happened relative to Event 2 (treatment initiation).
  - NOTE: the note AFTER paraphrasing. Wording, and sometimes structure, has changed.

Find the shortest contiguous span of the NOTE, copied VERBATIM (character-for-character,
including punctuation), that expresses the same temporal relationship between Event 1
and Event 2 as ORIGINAL_SPAN. Prefer a single contiguous span even if the paraphrase
also mentions the events elsewhere.

If no part of the NOTE still expresses that temporal relationship (the information was
lost, generalized away, or made ambiguous during paraphrasing), return null for "span".
Do not paraphrase, summarize, or shorten the span yourself: copy exact substrings only.

Return JSON only:
{"span": string | null, "confidence": number, "reason": string}"""

JUDGE_SYSTEM = """You are a strict annotator checking a proposed evidence span in isolation,
the same way an extraction system's "cited evidence" would be checked.

You are given: EVENT1 (diagnosis), EVENT2 (treatment initiation), the note's GOLD_RELATION,
the ORIGINAL_SPAN (pre-paraphrase wording, for reference only), and a CANDIDATE_SPAN taken
from the paraphrased note.

Judge the CANDIDATE_SPAN alone, as if it were the only evidence available:
  - relation_entailed: does the CANDIDATE_SPAN by itself state or unambiguously imply
    GOLD_RELATION between EVENT1 and EVENT2? (true/false)
  - same_information: does the CANDIDATE_SPAN convey the same temporal fact as the
    ORIGINAL_SPAN (allowing different wording), rather than something weaker, broader,
    or different? (true/false)

Return JSON only:
{"relation_entailed": true | false, "same_information": true | false, "notes": string}"""


def parse_json(content):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (content or "").strip(), flags=re.I | re.S)
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def chat(client, model, system, user, temperature, max_tokens=400):
    r = client.chat.completions.create(
        model=model, temperature=temperature, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return r.choices[0].message.content or ""


def relocate(clients, args, note_text, original_span, event1, event2, temperature, feedback=None):
    client, model = clients["relocate"]
    user = (f"Event 1 (diagnosis): {event1}\nEvent 2 (treatment initiation): {event2}\n\n"
            f"ORIGINAL_SPAN:\n{original_span}\n\nNOTE:\n{note_text}")
    if feedback:
        user += f"\n\nFeedback on the previous attempt: {feedback}"
    out = chat(client, model, RELOCATE_SYSTEM, user, temperature, 300)
    obj = parse_json(out)
    if not isinstance(obj, dict):
        # parsing failure, NOT the same as the model saying the evidence is gone;
        # caller must retry rather than conclude evidence_lost_in_paraphrase
        return "__PARSE_ERROR__", "invalid_json"
    span = obj.get("span")
    span = span if isinstance(span, str) and span.strip() else None
    return span, obj.get("reason", "")


def judge(clients, args, candidate_span, original_span, event1, event2, gold_relation):
    client, model = clients["judge"]
    user = (f"EVENT1 (diagnosis): {event1}\nEVENT2 (treatment initiation): {event2}\n"
            f"GOLD_RELATION: {gold_relation}\n\nORIGINAL_SPAN:\n{original_span}\n\n"
            f"CANDIDATE_SPAN:\n{candidate_span}")
    try:
        out = chat(client, model, JUDGE_SYSTEM, user, 0.0, 200)
    except Exception as exc:
        return {"relation_entailed": None, "same_information": None, "error": type(exc).__name__}
    obj = parse_json(out)
    if not isinstance(obj, dict):
        return {"relation_entailed": None, "same_information": None, "error": "invalid_json"}
    re_ = obj.get("relation_entailed")
    si = obj.get("same_information")
    return {"relation_entailed": re_ if isinstance(re_, bool) else None,
            "same_information": si if isinstance(si, bool) else None,
            "notes": obj.get("notes")}


def process_note(row, clients, args):
    note_id = str(row["note_id"])
    text = str(row["text"])
    original_span = row.get("gold_evidence_span")
    rec = {"v": CACHE_VERSION, "note_id": note_id}
    if not isinstance(original_span, str) or not original_span.strip():
        rec["reason"] = "no_original_span"
        return rec
    feedback = None
    rejections = []
    for attempt in range(1, args.max_attempts + 1):
        temp = 0.0 if attempt == 1 else min(0.7, 0.2 * (attempt - 1))
        try:
            span, reloc_reason = relocate(clients, args, text, original_span,
                                          row["event1"], row["event2"], temp, feedback)
        except Exception as exc:
            rejections.append({"attempt": attempt, "reason": f"relocate_error:{type(exc).__name__}"})
            continue
        if span == "__PARSE_ERROR__":
            feedback = "Your previous reply was not valid JSON. Return JSON only, exactly the schema given."
            rejections.append({"attempt": attempt, "reason": "invalid_json"})
            continue
        if span is None:
            rec.update({"reason": "evidence_lost_in_paraphrase", "relocator_reason": reloc_reason,
                       "attempts": attempt, "rejections": rejections})
            return rec
        if span not in text:
            feedback = ("Your span must be an EXACT substring of NOTE (character-for-character). "
                       f"This was not found verbatim: \"{span[:200]}\"")
            rejections.append({"attempt": attempt, "reason": "not_verbatim", "span": span})
            continue
        j = judge(clients, args, span, original_span, row["event1"], row["event2"], row["gold_relation"])
        if j["relation_entailed"] is None:
            rejections.append({"attempt": attempt, "reason": "judge_error", "span": span})
            continue
        if not j["relation_entailed"]:
            feedback = ("The judge said that span does not, on its own, entail the note's "
                       f"gold relation ({row['gold_relation']}). Find a span that does, or return null.")
            rejections.append({"attempt": attempt, "reason": "relation_not_entailed", "span": span,
                              "judge_notes": j.get("notes")})
            continue
        tier = "verified" if j["same_information"] else "relation_only"
        rec.update({"reason": "accepted", "gold_evidence_span_v2": span, "validity_tier": tier,
                   "judge_relation_entailed": j["relation_entailed"],
                   "judge_same_information": j["same_information"], "attempts": attempt,
                   "rejections": rejections})
        return rec
    rec.update({"reason": "attempts_exhausted", "attempts": args.max_attempts, "rejections": rejections})
    return rec


def make_clients(args):
    from openai import OpenAI
    reloc_client = OpenAI(base_url=args.api_base, api_key=args.api_key)
    judge_client = OpenAI(base_url=args.judge_api_base or args.api_base, api_key=args.api_key)
    return {"relocate": (reloc_client, args.model),
            "judge": (judge_client, args.judge_model or args.model)}


def test_patients(df, seed, n):
    pids = list(df["patient_id"].unique())
    random.Random(seed).shuffle(pids)
    return set(pids[:n])


def load_cache(path):
    cache = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("v") == CACHE_VERSION:
                        cache[r["note_id"]] = r
                except Exception:
                    pass
    return cache


def run(args, clients):
    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.notes_csv)
    need = {"note_id", "patient_id", "text", "gold_relation", "gold_evidence_span"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"notes_csv is missing columns: {sorted(missing)}")
    df = df[df["gold_relation"].notna()].reset_index(drop=True)
    if "event1" not in df.columns:
        df["event1"] = "the diagnosis"
    if "event2" not in df.columns:
        df["event2"] = "the treatment initiation"
    tset = test_patients(df, args.test_seed, args.test_n)

    cache_path = args.cache_path or os.path.join(args.out_dir, "gold_span_cache.jsonl")
    if args.debug_n:
        df = df.head(args.debug_n)
        cache_path = os.path.join(args.out_dir, "gold_span_cache_debug.jsonl")
        if os.path.exists(cache_path):
            os.remove(cache_path)
    elif args.limit:
        df = df.head(args.limit)
    cache = load_cache(cache_path)
    todo = [r for _, r in df.iterrows() if str(r["note_id"]) not in cache]
    print(f"{len(df)} notes, {len(cache)} cached, {len(todo)} to process")

    lock = threading.Lock()
    done = 0
    with open(cache_path, "a") as cf, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_note, r, clients, args): r for r in todo}
        for fut in as_completed(futs):
            rec = fut.result()
            with lock:
                cache[rec["note_id"]] = rec
                cf.write(json.dumps(rec) + "\n")
                cf.flush()
                done += 1
                if done % 25 == 0 or done == len(todo):
                    print(f"  processed {done}/{len(todo)}")

    out_rows, rej_rows = [], []
    reason_counts = Counter()
    tier_by_style = Counter()
    for _, base in df.iterrows():
        rec = cache.get(str(base["note_id"]))
        if not rec:
            continue
        reason_counts[rec.get("reason", "unknown")] += 1
        for rj in rec.get("rejections", []):
            rej_rows.append({"note_id": rec["note_id"], "style": base.get("style"),
                             "gold_relation": base["gold_relation"], **rj})
        row = {"note_id": rec["note_id"], "patient_id": base["patient_id"], "style": base.get("style"),
              "gold_relation": base["gold_relation"], "gold_evidence_span_orig": base["gold_evidence_span"],
              "text": base["text"], "reason": rec.get("reason"),
              "gold_evidence_span_v2": rec.get("gold_evidence_span_v2"),
              "validity_tier": rec.get("validity_tier"),
              "judge_relation_entailed": rec.get("judge_relation_entailed"),
              "judge_same_information": rec.get("judge_same_information"),
              "attempts": rec.get("attempts")}
        out_rows.append(row)
        if rec.get("validity_tier"):
            tier_by_style[(base.get("style"), rec["validity_tier"])] += 1

    out_df = pd.DataFrame(out_rows)
    out_path = os.path.join(args.out_dir, "gold_spans_v2.csv")
    out_df.to_csv(out_path, index=False)
    if not args.no_test_split:
        out_df[out_df["patient_id"].isin(tset)].to_csv(
            os.path.join(args.out_dir, "gold_spans_v2_test.csv"), index=False)
    pd.DataFrame(rej_rows).to_csv(os.path.join(args.out_dir, "gold_span_rejections.csv"), index=False)

    n = len(out_df)
    n_accepted = int((out_df["reason"] == "accepted").sum())
    report = {
        "n_notes": n,
        "relocate_model": args.model, "judge_model": args.judge_model or args.model,
        "reason_counts": dict(reason_counts),
        "accepted": n_accepted,
        "accepted_rate": round(n_accepted / max(1, n), 4),
        "validity_tier_counts": dict(Counter(out_df["validity_tier"].dropna())),
        "evidence_lost_in_paraphrase_rate": round(reason_counts.get("evidence_lost_in_paraphrase", 0) / max(1, n), 4),
        "mean_attempts_accepted": round(float(out_df.loc[out_df["reason"] == "accepted", "attempts"].mean()), 3) if n_accepted else None,
        "accepted_by_style_and_tier": {f"{s}|{t}": c for (s, t), c in tier_by_style.items()},
        "rows_in_seed7_test_patients": int(out_df["patient_id"].isin(tset).sum()),
    }
    with open(os.path.join(args.out_dir, "gold_spans_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

    accepted = out_df[out_df["reason"] == "accepted"]
    rng = random.Random(0)
    take = min(args.audit_n, len(accepted))
    audit = []
    for i in (rng.sample(range(len(accepted)), take) if take else []):
        r = accepted.iloc[i]
        audit.append({"note_id": r["note_id"], "validity_tier": r["validity_tier"],
                      "gold_evidence_span_orig": r["gold_evidence_span_orig"],
                      "gold_evidence_span_v2": r["gold_evidence_span_v2"], "note_text": r["text"],
                      "human_span_is_verbatim_in_note_yes_no": "",
                      "human_span_supports_gold_relation_yes_no": "",
                      "human_same_information_as_original_yes_no": "", "human_notes": ""})
    pd.DataFrame(audit).to_csv(os.path.join(args.out_dir, "gold_span_audit_sample.csv"), index=False)

    if args.debug_n and rej_rows:
        print("\n=== DEBUG: rejected attempts ===")
        for rj in rej_rows[:12]:
            print(f"\n[{rj['note_id']} | attempt {rj['attempt']} | {rj['reason']}]")
            if rj.get("span"):
                print("candidate:", str(rj["span"])[:300])
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--notes_csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--api_base", default="http://localhost:8000/v1")
    p.add_argument("--api_key", default=os.getenv("VLLM_API_KEY", "EMPTY"))
    p.add_argument("--model", default="mistralai/Mistral-Small-3.2-24B-Instruct-2506", help="relocator model")
    p.add_argument("--judge_model", default=None, help="validator model (recommended: different from the gate model)")
    p.add_argument("--judge_api_base", default=None)
    p.add_argument("--max_attempts", type=int, default=3)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="pilot: only the first N notes")
    p.add_argument("--debug_n", type=int, default=0, help="fresh run on the first N notes, printing rejections")
    p.add_argument("--audit_n", type=int, default=30)
    p.add_argument("--test_seed", type=int, default=7)
    p.add_argument("--test_n", type=int, default=150)
    p.add_argument("--no_test_split", action="store_true")
    p.add_argument("--cache_path", default=None)
    args = p.parse_args()
    run(args, make_clients(args))


if __name__ == "__main__":
    main()
