#!/usr/bin/env python3
"""
Judge.py - supplementary independent (cross-family) judge for the marker-free controls.


Only the standard library is required.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

LABELS = ["EVENT1-BEFORE-EVENT2", "EVENT2-BEFORE-EVENT1", "EVENT1-OVERLAP-EVENT2"]
PROMPT_VERSION = "v1-qwen-judge"

JUDGE_SYSTEM = """You are an auditor of clinical notes. You see ONE note and nothing else.
Event 1 is the working diagnosis. Event 2 is the initiation of a treatment.
Possible temporal relations between them:
- EVENT1-BEFORE-EVENT2: the diagnosis was made before the treatment was started.
- EVENT2-BEFORE-EVENT1: the treatment was started before the diagnosis was made.
- EVENT1-OVERLAP-EVENT2: the treatment was started during the diagnostic workup / at the same time.

Decide how much the note itself tells you about this relation. Rules:
1. Do NOT infer the relation from the order in which the events are mentioned.
2. Do NOT use prior knowledge about which relation is common or typical.
3. Answer with exactly one verdict:
   - "explicit": a temporal statement directly fixes the relation (a date, day number, ordinal,
     or temporal connective such as before / after / prior to / following / during / at the same time).
   - "implied": no direct temporal statement, but the relation can still be inferred from residual
     phrasing (for example causal or purpose wording such as "started X because of Y", "X for confirmed Y").
   - "none": the note does not let you determine the relation.
4. If the verdict is "explicit" or "implied", give the relation and copy, VERBATIM, the shortest
   contiguous span of the note that supports it. If "none", set relation and evidence to null.

Return ONLY a JSON object, no prose, no markdown:
{"verdict": "explicit|implied|none", "relation": "EVENT1-BEFORE-EVENT2|EVENT2-BEFORE-EVENT1|EVENT1-OVERLAP-EVENT2|null", "evidence": "...|null", "rationale": "one short sentence"}"""

SPAN_SYSTEM = """You see a short text span copied from a clinical note. Event 1 is the working diagnosis,
Event 2 is the initiation of a treatment. Using ONLY this span, which temporal relation does it entail?
Options: EVENT1-BEFORE-EVENT2, EVENT2-BEFORE-EVENT1, EVENT1-OVERLAP-EVENT2, or NONE if the span does not
entail any of them. Do not use mention order or prior knowledge.
Return ONLY JSON: {"relation": "EVENT1-BEFORE-EVENT2|EVENT2-BEFORE-EVENT1|EVENT1-OVERLAP-EVENT2|NONE", "rationale": "one short sentence"}"""

_lock = threading.Lock()


# ----------------------------------------------------------------------------- utilities
def norm_ws(s):
    return re.sub(r"\s+", " ", (s or "")).strip()


def norm_relation(x):
    if x is None:
        return None
    x = str(x).strip().upper().replace("_", "-")
    return x if x in LABELS else None


def norm_tier(x):
    if x is None:
        return "none"
    x = str(x).strip().lower().replace("-", "_")
    if x in ("strict",):
        return "strict"
    if x in ("implied", "implied_only"):
        return "implied_only"
    return "none"


def extract_json(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def call_chat(args, system, user):
    payload = {
        "model": args.model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
    }
    if args.no_think:
        # Honoured by vLLM / llama.cpp for Qwen3-style hybrid-thinking templates; ignored elsewhere.
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    if args.json_mode:
        payload["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", "none")},
    )
    last = None
    for attempt in range(args.retries):
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"request failed after {args.retries} attempts: {last}")


def cache_key(args, kind, text):
    h = hashlib.sha256()
    h.update("|".join([args.model, PROMPT_VERSION, kind, text]).encode("utf-8"))
    return h.hexdigest()


def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_system_prompt(args, default):
    if args.judge_prompt:
        with open(args.judge_prompt, encoding="utf-8") as f:
            return f.read()
    return default


# ----------------------------------------------------------------------------- judge
def judge_one(args, system, note_id, condition, text):
    raw = call_chat(args, system, "NOTE:\n" + text)
    j = extract_json(raw) or {}
    verdict = str(j.get("verdict", "")).strip().lower()
    if verdict not in ("explicit", "implied", "none"):
        verdict = "parse_error"
    relation = norm_relation(j.get("relation"))
    evidence = j.get("evidence")
    evidence = None if evidence in (None, "null", "") else str(evidence)
    evidence_ok = bool(evidence) and norm_ws(evidence).lower() in norm_ws(text).lower()
    # A determination without a verbatim, in-note span is reported as "unsupported" and
    # excluded; counting it as "none" would inflate the strict tier.
    if verdict in ("explicit", "implied") and (relation is None or not evidence_ok):
        verdict_eff, relation_eff = "unsupported", None
    elif verdict == "none":
        verdict_eff, relation_eff = "none", None
    else:
        verdict_eff, relation_eff = verdict, relation
    return {
        "note_id": note_id, "condition": condition,
        "verdict_raw": verdict, "relation_raw": relation, "evidence": evidence, "evidence_ok": evidence_ok,
        "verdict": verdict_eff, "relation": relation_eff,
        "rationale": j.get("rationale"), "raw": raw,
    }


def cmd_judge(args):
    system = load_system_prompt(args, JUDGE_SYSTEM)
    notes = load_jsonl(args.input)

    tasks = []
    for r in notes:
        for cond in ("clean", "silent_removal", "silent_keep"):
            txt = r.get(getattr(args, "f_" + cond))
            if txt:
                tasks.append((str(r[args.f_id]), cond, txt))

    cache = {}
    if os.path.exists(args.out):
        for r in load_jsonl(args.out):
            if "key" in r:
                cache[r["key"]] = r
    todo = [t for t in tasks if cache_key(args, t[1], t[2]) not in cache]
    print(f"{len(tasks)} judgments needed, {len(tasks) - len(todo)} cached, {len(todo)} to run", file=sys.stderr)

    with open(args.out, "a", encoding="utf-8") as fout, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(judge_one, args, system, *t): t for t in todo}
        done = 0
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                res = fut.result()
            except Exception as e:  # keep going; failures are listed at the end
                print(f"FAILED {t[0]}/{t[1]}: {e}", file=sys.stderr)
                continue
            res["key"] = cache_key(args, t[1], t[2])
            res["model"] = args.model
            res["prompt_version"] = PROMPT_VERSION
            with _lock:
                fout.write(json.dumps(res, ensure_ascii=False) + "\n")
                fout.flush()
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(todo)}", file=sys.stderr)
    print("done ->", args.out, file=sys.stderr)


# ----------------------------------------------------------------------------- report
def qwen_tier(j):
    """Map a judgment of an edited (removal) note to a tier."""
    if j is None or j["verdict"] in ("parse_error", "unsupported"):
        return "missing"
    if j["verdict"] == "none" or j["relation"] is None:
        return "strict"
    if j["verdict"] == "implied":
        return "implied_only"
    return "rejected"          # explicit residual temporal statement -> not a valid removal


def cohen_kappa(pairs):
    n = len(pairs)
    if n == 0:
        return float("nan")
    cats = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def cmd_report(args):
    notes = {str(r[args.f_id]): r for r in load_jsonl(args.input)}
    J = {}
    for r in load_jsonl(args.judgments):
        J[(r["note_id"], r["condition"])] = r
    os.makedirs(args.outdir, exist_ok=True)

    sets = {k: [] for k in [
        "prefilter_qwen_clean_ok",
        "removal_strict_qwen", "removal_strict_both_judges",
        "removal_implied_qwen", "removal_implied_both_judges",
        "removal_all_qwen",           # strict + implied_only, Qwen tiers, clean recovered by Qwen
        "keep_valid_qwen",
    ]}
    conf = Counter()
    pairs = []
    n_clean = n_clean_ok = 0

    for nid, r in notes.items():
        gold = norm_relation(r[args.f_gold])
        jc = J.get((nid, "clean"))
        clean_ok = bool(jc and jc["relation"] == gold)
        if jc:
            n_clean += 1
            n_clean_ok += clean_ok
        if clean_ok:
            sets["prefilter_qwen_clean_ok"].append(nid)

        jr = J.get((nid, "silent_removal"))
        if jr is not None:
            qt = qwen_tier(jr)
            ot = norm_tier(r.get(args.f_orig_tier))
            conf[(ot, qt)] += 1
            if ot in ("strict", "implied_only") and qt in ("strict", "implied_only", "rejected"):
                pairs.append((ot, qt))
            if clean_ok:
                if qt == "strict":
                    sets["removal_strict_qwen"].append(nid)
                    if ot == "strict":
                        sets["removal_strict_both_judges"].append(nid)
                if qt == "implied_only":
                    sets["removal_implied_qwen"].append(nid)
                    if ot == "implied_only":
                        sets["removal_implied_both_judges"].append(nid)
                if qt in ("strict", "implied_only"):
                    sets["removal_all_qwen"].append(nid)

        jk = J.get((nid, "silent_keep"))
        if jk is not None and clean_ok and jk["relation"] == gold:
            sets["keep_valid_qwen"].append(nid)

    for k, v in sets.items():
        with open(os.path.join(args.outdir, k + ".json"), "w") as f:
            json.dump(sorted(v), f)

    # Reference ceiling: how well does Mistral agree with ITS OWN re-run? (field optional)
    rr_pairs, qr_pairs = [], []
    for nid, rec in notes.items():
        rt, ot = rec.get("mistral_rerun_tier"), norm_tier(rec.get(args.f_orig_tier))
        if rt and ot in ("strict", "implied_only"):
            rr_pairs.append((ot, norm_tier(rt)))
            jr = J.get((nid, "silent_removal"))
            if jr is not None and qwen_tier(jr) in ("strict", "implied_only", "rejected"):
                qr_pairs.append((norm_tier(rt), qwen_tier(jr)))
    n_agree = sum(a == b for a, b in pairs)
    summary = {
        "judge_model": next(iter(J.values()))["model"] if J else None,
        "prompt_version": PROMPT_VERSION,
        "clean_notes_judged": n_clean,
        "clean_recovered_by_qwen": n_clean_ok,
        "set_sizes": {k: len(v) for k, v in sets.items()},
        "tier_agreement_n": len(pairs),
        "tier_percent_agreement": (n_agree / len(pairs)) if pairs else None,
        "tier_cohens_kappa": cohen_kappa(pairs),
        "mistral_vs_mistral_rerun": {"n": len(rr_pairs),
            "percent_agreement": (sum(a == b for a, b in rr_pairs) / len(rr_pairs)) if rr_pairs else None,
            "cohens_kappa": cohen_kappa(rr_pairs)},
        "qwen_vs_mistral_rerun": {"n": len(qr_pairs),
            "percent_agreement": (sum(a == b for a, b in qr_pairs) / len(qr_pairs)) if qr_pairs else None,
            "cohens_kappa": cohen_kappa(qr_pairs)},
        "confusion_orig_vs_qwen": {f"{o}->{q}": c for (o, q), c in sorted(conf.items())},
        "interpretation_note": (
            "orig 'none' = no accepted Mistral edit. 'rejected' = Qwen finds an explicit temporal "
            "statement left in the edit. Agreement/kappa computed only where both judges assign a tier "
            "(strict/implied_only vs strict/implied_only/rejected)."),
    }
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print("\nNext: score both gates on these ID lists with existing predictions (no re-run needed):\n"
          f"  python qwen_subset_eval.py --idlists {args.outdir} --notes ... --llm_dir ... --bert_dir ...")


# ----------------------------------------------------------------------------- spans
def cmd_spans(args):
    system = load_system_prompt(args, SPAN_SYSTEM)
    rows = load_jsonl(args.input)
    out, ok, n = [], 0, 0

    def one(r):
        raw = call_chat(args, system, "SPAN:\n" + r["span"])
        j = extract_json(raw) or {}
        rel = norm_relation(j.get("relation"))
        return {"note_id": str(r[args.f_id]), "gold": r[args.f_gold], "pred": rel,
                "entails_gold": rel == norm_relation(r[args.f_gold]), "rationale": j.get("rationale")}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(one, rows):
            out.append(res)
            n += 1
            ok += res["entails_gold"]
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"span entails gold: {ok}/{n} = {ok / n:.3f}" if n else "no spans")


# ----------------------------------------------------------------------------- cli
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, serve=True):
        sp.add_argument("--input", required=True)
        sp.add_argument("--f-id", default="note_id")
        sp.add_argument("--f-gold", default="gold")
        if serve:
            sp.add_argument("--base-url", default="http://localhost:8000/v1")
            sp.add_argument("--model", required=True, help="exact model name your server exposes")
            sp.add_argument("--workers", type=int, default=8)
            sp.add_argument("--seed", type=int, default=7)
            sp.add_argument("--max-tokens", type=int, default=400)
            sp.add_argument("--timeout", type=int, default=120)
            sp.add_argument("--retries", type=int, default=4)
            sp.add_argument("--no-think", action="store_true", default=True,
                            help="disable hybrid thinking where the template supports it (default on)")
            sp.add_argument("--json-mode", action="store_true", help="send response_format=json_object")
            sp.add_argument("--judge-prompt", default=None, help="file overriding the built-in system prompt")

    sj = sub.add_parser("judge")
    common(sj)
    sj.add_argument("--out", required=True)
    sj.add_argument("--f-clean", dest="f_clean", default="clean")
    sj.add_argument("--f-silent_removal", dest="f_silent_removal", default="silent_removal")
    sj.add_argument("--f-silent_keep", dest="f_silent_keep", default="silent_keep")
    sj.set_defaults(fn=cmd_judge)

    sr = sub.add_parser("report")
    common(sr, serve=False)
    sr.add_argument("--judgments", required=True)
    sr.add_argument("--outdir", required=True)
    sr.add_argument("--f-orig-tier", dest="f_orig_tier", default="orig_removal_tier")
    sr.set_defaults(fn=cmd_report)

    ss = sub.add_parser("spans")
    common(ss)
    ss.add_argument("--out", required=True)
    ss.set_defaults(fn=cmd_spans)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
