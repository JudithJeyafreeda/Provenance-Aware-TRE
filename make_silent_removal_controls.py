
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

REMOVE_SYSTEM = """You edit synthetic clinical notes for a controlled experiment.
Rewrite the note so that a reader can no longer tell whether the diagnosis (Event 1)
happened before, after, or at the same time as the treatment initiation (Event 2).

Rules:
1. Keep BOTH events mentioned in the note.
2. Delete every day number, date, ordinal day (e.g. "first day"), and every temporal
   connective or cue (before, after, prior to, following, subsequently, then, while,
   during, concurrently, at the same time, later, earlier, initially, once, upon, etc.).
   Do not let sentence order or bullet order imply the sequence: state the two events
   together in one neutral clause or list, joined with "and".
3. Do NOT add any statement that information is missing, unclear, omitted, unknown or
   unspecified. No brackets, placeholders, ellipses, or notes about the edit. The result
   must read like a normal note that simply does not give the timing.
4. ALWAYS mention the diagnosis (Event 1) FIRST and the treatment initiation (Event 2)
   SECOND, whatever the original order was. The order of mention must not carry the answer.
5. Keep everything else (style, formatting, service, unrelated content) unchanged.
   Do not add new facts.
Return ONLY the edited note text."""

AGGRESSIVE_SUFFIX = """

Be more aggressive than in the previous attempt. Also remove causal, purpose and outcome
phrasing (for example "to manage", "as a result", "in response to", "which was treated with",
"correlating with", "developed") and state the two events as a plain list, for example
"Diagnosis: ...; treatment: ...", so no order can be inferred."""

KEEP_SYSTEM = """You make a MINIMAL edit to a synthetic clinical note for a controlled experiment.
Change only a few words (synonyms or small rephrasing) and keep the structure, style and
formatting. Preserve every clinical fact, every day number, and the exact temporal
relationship between the diagnosis (Event 1) and the treatment initiation (Event 2).
Do not add or remove information and do not add comments.
Return ONLY the edited note text."""

JUDGE_SYSTEM = """You are a strict annotator. Decide whether the note states or unambiguously
implies the temporal relationship between Event 1 (the diagnosis) and Event 2 (the
treatment initiation).

Evidence counts only if it is explicit: a day number or date, an ordinal day, or an
ordering / co-occurrence word or phrase. Sentence order alone is NOT evidence.

Relations:
- EVENT1-BEFORE-EVENT2: the diagnosis occurred before the treatment was initiated.
- EVENT2-BEFORE-EVENT1: the treatment was initiated before the diagnosis was made.
- EVENT1-OVERLAP-EVENT2: the diagnosis and treatment occurred concurrently / on the same day.

Return JSON only:
{"determinable": true | false,
 "relation": "EVENT1-BEFORE-EVENT2" | "EVENT2-BEFORE-EVENT1" | "EVENT1-OVERLAP-EVENT2" | null,
 "event1_mentioned": true | false,
 "event2_mentioned": true | false,
 "evidence": string | null}
If determinable is false, relation must be null."""

MARKER_PATTERNS = {
    "omitted": r"\bomitt?ed\b",
    "not_stated": r"\bnot (?:specified|documented|stated|recorded|available|clear|reported|provided|known|determined)\b",
    "unspecified": r"\b(?:unspecified|undocumented|undated|unknown|unclear|unavailable)\b",
    "cannot_be": r"\b(?:cannot|can not|unable to|not possible to)\b",
    "redacted": r"\b(?:redact\w*|removed|missing|masked)\b",
    "no_info": r"\bno (?:information|details?|data|record)\b",
    "meta_temporal": r"\b(?:temporal|timing|chronolog\w*|relationship)\b",
    "brackets": r"\[[^\]]*\]",
    "ellipsis": r"\.\.\.|\u2026",
    "filler": r"\*\*\*|xxx+|___+|\bTBD\b|\bn/?a\b",
}


CACHE_VERSION = 2


def marker_hits(text):
    return {k for k, p in MARKER_PATTERNS.items() if re.search(p, text, flags=re.I)}


EXPLICIT_CUE = re.compile(
    r"\b(?:day|hospital day|hd|pod|d)\s*#?\s*\d+\b"
    r"|\b(?:first|second|third|fourth|fifth|sixth|seventh|next|following|previous|prior)\s+(?:day|morning|evening|night)\b"
    r"|\b(?:today|yesterday|tomorrow|overnight|on admission|upon admission|at admission)\b"
    r"|\b(?:before|after|prior to|following|subsequent(?:ly)?|thereafter|afterwards?|then|while|during|"
    r"concurrent(?:ly)?|simultaneous(?:ly)?|at the same time|same day|later|earlier|initially|once|upon|"
    r"previously|preceded|followed|as a result|consequently|in response to|ensuing|meanwhile|"
    r"at that time|shortly)\b"
    r"|\b(?:pre|post)[- ]?(?:op(?:erative(?:ly)?)?|treatment|procedure|intervention)\b"
    r"|\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b",
    flags=re.I)


def explicit_cues(text):
    return [m.group(0) for m in EXPLICIT_CUE.finditer(text)]


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def novel_frac(orig, edited):
    o = set(tokens(orig))
    e = tokens(edited)
    return sum(t not in o for t in e) / max(1, len(e))


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


def chat(client, model, system, user, temperature, max_tokens=700):
    r = client.chat.completions.create(
        model=model, temperature=temperature, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return r.choices[0].message.content or ""


def event_block(event1, event2):
    return f"Event 1 (diagnosis): {event1}\nEvent 2 (treatment initiation): {event2}\n\n"


def judge(clients, args, text, event1, event2):
    client, model = clients["judge"]
    try:
        out = chat(client, model, JUDGE_SYSTEM, event_block(event1, event2) + "Note:\n" + text, 0.0, 250)
    except Exception as exc:  # network / server error
        return {"determinable": None, "relation": None, "event1_mentioned": None,
                "event2_mentioned": None, "error": type(exc).__name__}
    obj = parse_json(out)
    if not isinstance(obj, dict):
        return {"determinable": None, "relation": None, "event1_mentioned": None,
                "event2_mentioned": None, "error": "invalid_json"}
    rel = obj.get("relation") if obj.get("relation") in RELATIONS else None
    det = obj.get("determinable")
    det = det if isinstance(det, bool) else None
    if det is False:
        rel = None
    return {"determinable": det, "relation": rel,
            "event1_mentioned": obj.get("event1_mentioned"),
            "event2_mentioned": obj.get("event2_mentioned"),
            "evidence": obj.get("evidence")}


def edit(clients, args, mode, text, event1, event2, temperature, attempt=1, feedback=None):
    client, model = clients["edit"]
    system = REMOVE_SYSTEM if mode == "silent_removal" else KEEP_SYSTEM
    if mode == "silent_removal" and attempt >= 2:
        system += AGGRESSIVE_SUFFIX
    user = event_block(event1, event2) + "Note:\n" + text
    if feedback and mode == "silent_removal":
        user += (f"\n\nFeedback on the previous attempt: a reader could still work out the timing "
                 f"from this wording: \"{feedback}\". Remove or neutralise it.")
    out = chat(client, model, system, user, temperature, 900)
    out = out.strip()
    out = re.sub(r"^```(?:\w+)?\s*|\s*```$", "", out).strip()
    out = re.sub(r"^(?:edited|rewritten)?\s*note\s*:\s*", "", out, flags=re.I).strip()
    return out


def try_variant(clients, args, mode, row, orig_text):
    """Return (accepted_record | None, reasons, rejections)."""
    gold = row["gold_relation"]
    reasons, rejections = [], []
    feedback = None
    best_implied = None

    def reject(attempt, reason, text, evidence=None):
        reasons.append(reason)
        rejections.append({"attempt": attempt, "reason": reason, "text": (text or "")[:1500],
                           "judge_evidence": evidence})

    for attempt in range(1, args.max_attempts + 1):
        temp = 0.0 if attempt == 1 else min(0.9, 0.3 * (attempt - 1) + args.edit_temperature)
        try:
            edited = edit(clients, args, mode, orig_text, row["event1"], row["event2"], temp,
                          attempt=attempt, feedback=feedback)
        except Exception as exc:
            reasons.append(f"edit_error:{type(exc).__name__}")
            continue
        if not edited or edited.strip() == orig_text.strip():
            reject(attempt, "empty_or_unchanged", edited)
            continue
        ratio = len(edited) / max(1, len(orig_text))
        lo, hi = (args.remove_len_lo, args.remove_len_hi) if mode == "silent_removal" else (0.70, 1.40)
        if not lo <= ratio <= hi:
            reject(attempt, "length_ratio", edited)
            continue
        new_markers = marker_hits(edited) - marker_hits(orig_text)
        if new_markers:
            reject(attempt, "marker:" + "+".join(sorted(new_markers)), edited)
            continue
        nf = novel_frac(orig_text, edited)
        max_nf = args.max_novel_remove if mode == "silent_removal" else args.max_novel_keep
        if nf > max_nf:
            reject(attempt, "novel_tokens", edited)
            continue
        cues = explicit_cues(edited) if mode == "silent_removal" else []
        if cues:
            feedback = cues[0]
            reject(attempt, "explicit_cue_remains", edited, ", ".join(cues[:5]))
            continue
        j = judge(clients, args, edited, row["event1"], row["event2"])
        if j["determinable"] is None:
            reasons.append("judge_error")
            continue
        if not (j.get("event1_mentioned") is True and j.get("event2_mentioned") is True):
            reject(attempt, "event_not_mentioned", edited)
            continue
        ev = j.get("evidence") if isinstance(j.get("evidence"), str) else None
        rec = {"text": edited, "attempts": attempt, "novel_token_frac": round(nf, 4),
               "length_ratio": round(ratio, 4), "judge_edit_determinable": j["determinable"],
               "judge_edit_relation": j["relation"], "judge_evidence": ev}
        if mode == "silent_removal":
            if j["determinable"] is False:
                rec["validity_tier"] = "strict"
                return rec, reasons, rejections
            # no explicit cue left, but the judge still infers an order from phrasing
            if args.strict_only:
                feedback = ev
                reject(attempt, "still_determinable", edited, ev)
                continue
            rec["validity_tier"] = "implied_only"
            best_implied = rec
            feedback = ev
            reject(attempt, "implied_timing_retry", edited, ev)
            continue
        if not (j["determinable"] is True and j["relation"] == gold):
            reject(attempt, "relation_not_preserved", edited, ev)
            continue
        rec["validity_tier"] = "keep_verified"
        return rec, reasons, rejections
    return best_implied, reasons, rejections


def process_note(row, clients, args, prev=None):
    orig = str(row["text"])
    rec = {"v": CACHE_VERSION, "note_id": str(row["note_id"]), "variants": {}, "reasons": {}, "rejections": {}}
    if prev and "orig_valid" in prev:
        rec["judge_orig_determinable"] = prev.get("judge_orig_determinable")
        rec["judge_orig_relation"] = prev.get("judge_orig_relation")
        rec["orig_valid"] = bool(prev["orig_valid"])
    else:
        j0 = judge(clients, args, orig, row["event1"], row["event2"])
        rec["judge_orig_determinable"] = j0["determinable"]
        rec["judge_orig_relation"] = j0["relation"]
        rec["orig_valid"] = bool(j0["determinable"] is True and j0["relation"] == row["gold_relation"]
                                 and j0.get("event1_mentioned") is True and j0.get("event2_mentioned") is True)
    if not rec["orig_valid"]:
        return rec
    for mode in args.variants:
        acc, reasons, rejections = try_variant(clients, args, mode, row, orig)
        rec["reasons"][mode] = reasons
        rec["rejections"][mode] = rejections
        if acc:
            rec["variants"][mode] = acc
    return rec


def make_clients(args):
    from openai import OpenAI
    edit_client = OpenAI(base_url=args.api_base, api_key=args.api_key)
    judge_client = OpenAI(base_url=args.judge_api_base or args.api_base, api_key=args.api_key)
    return {"edit": (edit_client, args.model),
            "judge": (judge_client, args.judge_model or args.model)}


def test_patients(df, seed, n):
    b = df[df["gold_relation"].notna()]
    if "gold_evidence_span" in b.columns:
        b = b[b["gold_evidence_span"].notna()]
    pids = list(b["patient_id"].unique())
    random.Random(seed).shuffle(pids)
    return set(pids[:n])


def build_row(base, mode, acc):
    d = base.to_dict()
    d["text"] = acc["text"]
    d["original_text"] = base["text"]
    d["base_note_id"] = str(base["note_id"])
    d["note_id"] = f"{base['note_id']}_{mode}"
    d["example_id"] = d["note_id"]
    d["variant"] = mode
    d["evidence_case"] = mode
    d["evidence_available_oracle"] = 0 if mode == "silent_removal" else 1
    d["contradiction_present_oracle"] = 0
    d["removed_gold_span"] = False
    d["masked_cue"] = False
    d["injected_spans"] = ""
    d["contradiction_types"] = ""
    for k in ("judge_edit_determinable", "judge_edit_relation", "judge_evidence", "attempts",
              "novel_token_frac", "length_ratio", "validity_tier"):
        d[k] = acc.get(k)
    return d


def load_cache(path):
    cache = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    cache[r["note_id"]] = r
                except Exception:
                    pass
    return cache


def run(args, clients):
    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.notes_csv)
    need = {"note_id", "patient_id", "text", "gold_relation"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"notes_csv is missing columns: {sorted(missing)}")
    df = df[df["gold_relation"].notna()].reset_index(drop=True)
    if "event1" not in df.columns:
        df["event1"] = "the patient was diagnosed"
    if "event2" not in df.columns:
        df["event2"] = "treatment was initiated"
    tset = test_patients(df, args.test_seed, args.test_n)

    cache_path = args.cache_path or os.path.join(args.out_dir, "silent_cache.jsonl")
    if args.debug_n:
        df = df.head(args.debug_n)
        cache_path = os.path.join(args.out_dir, "silent_cache_debug.jsonl")
        if os.path.exists(cache_path):
            os.remove(cache_path)
    elif args.limit:
        df = df.head(args.limit)
    cache = load_cache(cache_path)

    # records from an older script version: reuse the judgment of the original note only
    prev = {k: v for k, v in cache.items() if v.get("v") != CACHE_VERSION}
    cache = {k: v for k, v in cache.items() if v.get("v") == CACHE_VERSION}
    todo = [r for _, r in df.iterrows() if str(r["note_id"]) not in cache
            or (cache[str(r["note_id"])].get("orig_valid")
                and any(v not in cache[str(r["note_id"])].get("reasons", {}) for v in args.variants))]
    print(f"{len(df)} notes, {len(cache)} cached (current version), "
          f"{len(prev)} older records (original-note judgments reused), {len(todo)} to process")

    lock = threading.Lock()
    done = 0
    with open(cache_path, "a") as cf, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_note, r, clients, args, prev.get(str(r["note_id"]))): r for r in todo}
        for fut in as_completed(futs):
            rec = fut.result()
            with lock:
                cache[rec["note_id"]] = rec
                cf.write(json.dumps(rec) + "\n")
                cf.flush()
                done += 1
                if done % 25 == 0 or done == len(todo):
                    print(f"  processed {done}/{len(todo)}")

    rows = {m: [] for m in args.variants}
    reason_counts = {m: Counter() for m in args.variants}
    rej_rows = []
    n_orig_valid = 0
    for _, base in df.iterrows():
        rec = cache.get(str(base["note_id"]))
        if not rec:
            continue
        n_orig_valid += int(rec.get("orig_valid", False))
        for m in args.variants:
            if m in rec.get("variants", {}):
                rows[m].append(build_row(base, m, rec["variants"][m]))
            for r in rec.get("reasons", {}).get(m, []):
                reason_counts[m][r if r.startswith("marker") else r.split(":")[0]] += 1
            for rj in rec.get("rejections", {}).get(m, []):
                rej_rows.append({"note_id": rec["note_id"], "variant": m, "style": base.get("style"),
                                 "gold_relation": base["gold_relation"], "attempt": rj["attempt"],
                                 "reason": rj["reason"], "judge_evidence": rj["judge_evidence"],
                                 "original_text": base["text"], "edited_text": rj["text"]})
    pd.DataFrame(rej_rows).to_csv(os.path.join(args.out_dir, "silent_rejections.csv"), index=False)

    report = {"n_input_notes": int(len(df)), "n_original_judged_valid": int(n_orig_valid),
              "original_valid_rate": round(n_orig_valid / max(1, len(df)), 4),
              "edit_model": args.model, "judge_model": args.judge_model or args.model,
              "variants": {}}
    audit = []
    rng = random.Random(0)
    for m in args.variants:
        out_df = pd.DataFrame(rows[m])
        path = os.path.join(args.out_dir, f"{m}_controls.csv")
        out_df.to_csv(path, index=False)
        in_test = out_df["patient_id"].isin(tset) if len(out_df) else pd.Series(dtype=bool)
        if len(out_df) and not args.no_test_split:
            out_df[in_test].to_csv(os.path.join(args.out_dir, f"{m}_controls_test.csv"), index=False)
        report["variants"][m] = {
            "accepted": int(len(out_df)),
            "accepted_of_valid_originals": round(len(out_df) / max(1, n_orig_valid), 4),
            "validity_tiers": dict(Counter(out_df["validity_tier"])) if len(out_df) else {},
            "accepted_by_style": dict(Counter(out_df["style"])) if len(out_df) else {},
            "mean_attempts": round(float(out_df["attempts"].mean()), 3) if len(out_df) else None,
            "mean_novel_token_frac": round(float(out_df["novel_token_frac"].mean()), 4) if len(out_df) else None,
            "rows_in_seed7_test_patients": int(in_test.sum()) if len(out_df) else 0,
            "rejection_reasons_over_all_attempts": dict(reason_counts[m]),
        }
        print(f"{m}: accepted {len(out_df)} -> {path}")
        take = min(args.audit_n, len(out_df))
        for i in (rng.sample(range(len(out_df)), take) if take else []):
            r = out_df.iloc[i]
            audit.append({"example_id": r["example_id"], "variant": m, "validity_tier": r["validity_tier"],
                          "original_text": r["original_text"], "edited_text": r["text"],
                          "gold_relation": r["gold_relation"],
                          "judge_edit_determinable": r["judge_edit_determinable"],
                          "human_timing_determinable_yes_no": "",
                          "human_marker_or_omission_phrase_yes_no": "",
                          "human_mention_order_reveals_relation_yes_no": "",
                          "human_notes": ""})
    pd.DataFrame(audit).to_csv(os.path.join(args.out_dir, "silent_controls_audit_sample.csv"), index=False)
    with open(os.path.join(args.out_dir, "silent_controls_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

    if args.debug_n and rej_rows:
        print("\n=== DEBUG: first rejected edits ===")
        for rj in rej_rows[:12]:
            print(f"\n[{rj['note_id']} | {rj['variant']} | attempt {rj['attempt']} | {rj['reason']}]")
            if rj["judge_evidence"]:
                print("judge evidence:", rj["judge_evidence"])
            print("EDITED:", str(rj["edited_text"])[:500])
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--notes_csv", required=True, help="relation-eligible notes (needs note_id, patient_id, text, gold_relation; event1/event2 recommended)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--variants", default="silent_removal,silent_keep")
    p.add_argument("--api_base", default="http://localhost:8000/v1")
    p.add_argument("--api_key", default=os.getenv("VLLM_API_KEY", "EMPTY"))
    p.add_argument("--model", default="mistralai/Mistral-Small-3.2-24B-Instruct-2506", help="editor model")
    p.add_argument("--judge_model", default=None, help="validator model (recommended: different from the gate model)")
    p.add_argument("--judge_api_base", default=None)
    p.add_argument("--edit_temperature", type=float, default=0.2)
    p.add_argument("--max_attempts", type=int, default=5)
    p.add_argument("--remove_len_lo", type=float, default=0.30)
    p.add_argument("--remove_len_hi", type=float, default=1.25)
    p.add_argument("--max_novel_remove", type=float, default=0.25)
    p.add_argument("--max_novel_keep", type=float, default=0.35)
    p.add_argument("--strict_only", action="store_true",
                   help="drop the implied_only tier (judge still infers an order); default keeps it, labelled")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="pilot: only the first N notes")
    p.add_argument("--debug_n", type=int, default=0, help="fresh run on the first N notes, printing rejected edits")
    p.add_argument("--audit_n", type=int, default=30, help="audit rows per variant")
    p.add_argument("--test_seed", type=int, default=7)
    p.add_argument("--test_n", type=int, default=150)
    p.add_argument("--no_test_split", action="store_true")
    p.add_argument("--cache_path", default=None)
    args = p.parse_args()
    args.variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    bad = set(args.variants) - {"silent_removal", "silent_keep"}
    if bad:
        raise ValueError(f"unknown variants: {sorted(bad)}")
    run(args, make_clients(args))


if __name__ == "__main__":
    main()
