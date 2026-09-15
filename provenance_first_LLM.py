"""
Provenance-First Clinical NLP: Proof-of-Concept via LLM API (vLLM / OpenAI-compatible)
=========================================================================================

Requirements:
    pip install openai pandas numpy scipy networkx

file dependency: provenance_temporal_kg.py must be in the same
directory (or importable on PYTHONPATH).

Usage
-----
    python provenance_first_poc_llm.py \\
        --data_dir "/path/to/data" --seed 7 \\
        --api_base "path/to/api/base" \\
        --llm_model "model_name" \\
        --n_test 60 --n_manip_pairs 60 \\
        --min_coverage_chars 10 --min_coverage_frac 0.03 \\
        --necessity_n_probes 3 --necessity_agreement_threshold 0.5 \\
        --run_certification --cert_n_examples 20 --cert_n_samples 100 --cert_alpha 0.05 \\
        --site_id my_llm_site_v7

--data_dir must point at a directory already populated by
synthetic_longitudinal_generator_v1.py (synthetic_notes_v1.csv and,
optionally, synthetic_adversarial_v1.csv). Run that generator once; do not
regenerate it before each v7 run -- re-running it produces a different
random corpus and breaks the identical-split guarantee with any v6 run
you want to compare against.

Note on certification sample size / API cost: with the multi-probe
necessity check, each gated call now costs (1 + necessity_n_probes) LLM
calls instead of 2. A certification run with cert_n_examples=20 and
cert_n_samples=100 therefore costs roughly
    20 * 100 * (1 + necessity_n_probes) API calls for provenance_first
alone -- at necessity_n_probes=3 that is ~8,000 calls. Reduce
cert_n_examples first if this is too expensive; do not reduce
cert_n_samples below 100, since that reintroduces the exact ceiling
problem described above.
"""

import os
import re
import json
import random
import argparse
import time
import pickle
import difflib
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats
from openai import OpenAI

from provenance_temporal_kg import (
    ProvenanceTemporalKG, ingest_llm_result, aggregate_temporal_motifs,
)

RELATION_LABELS = ["EVENT1-BEFORE-EVENT2", "EVENT2-BEFORE-EVENT1", "EVENT1-OVERLAP-EVENT2"]


# ----------------------------------------------------------------------------
# 1. LLM client and structured prompting
# ----------------------------------------------------------------------------

RELATION_SCHEMA = {
    "type": "object",
    "properties": {
        "relation": {"type": "string", "enum": RELATION_LABELS},
        "evidence_spans": {
            "type": "array",
            "items": {"type": "string"},
            "description": "One or more VERBATIM substrings copied exactly from the note "
                            "that justify the relation.",
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["relation", "evidence_spans", "confidence"],
}

SYSTEM_PROMPT = (
    "You are a clinical temporal-relation extraction system. Given a clinical "
    "note describing a diagnosis event and a treatment event, determine their "
    "temporal relation and cite the exact evidence.\n\n"
    "Relation definitions:\n"
    "- EVENT1-BEFORE-EVENT2: the diagnosis occurred before the treatment was initiated.\n"
    "- EVENT2-BEFORE-EVENT1: the treatment was initiated before the diagnosis was made.\n"
    "- EVENT1-OVERLAP-EVENT2: the diagnosis and treatment occurred concurrently / during "
    "the same workup.\n\n"
    "You MUST cite evidence_spans as VERBATIM substrings copied exactly, character-for-"
    "character, from the note text -- do not paraphrase or summarize.\n\n"
    "Example:\n"
    'Note: "On day 3, the patient was diagnosed with pneumonia. Antibiotics were started on day 5."\n'
    'Correct evidence_spans: ["the patient was diagnosed with pneumonia", "Antibiotics were started on day 5"]\n'
    'Incorrect, do NOT do this (paraphrased): ["diagnosis of pneumonia occurred", "antibiotics given later"]\n\n'
    "If you cannot find clear textual evidence for your answer, still provide your best guess, "
    "but keep the evidence_spans as short and precise as possible (do not cite unrelated sentences).\n\n"
    "Respond with ONLY a single JSON object matching the required schema. Do not include "
    "any explanation, reasoning, or markdown code fences before or after the JSON."
)


def build_user_prompt(text):
    return f"Clinical note:\n\"\"\"\n{text}\n\"\"\"\n\nDetermine the temporal relation and cite evidence."


def _strip_code_fences(s):
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    first_brace = s.find("{")
    last_brace = s.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        s = s[first_brace:last_brace + 1]
    return s.strip()


def _extract_text_from_message(message):
    content = getattr(message, "content", None)
    if content and content.strip():
        return content, "content"
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning and reasoning.strip():
        return reasoning, "reasoning_content"
    return None, None


def _one_api_call(client, model, messages, temperature, max_tokens, timeout, structured_mode):
    kwargs = dict(model=model, messages=messages, temperature=temperature,
                  max_tokens=max_tokens, timeout=timeout)
    if structured_mode == "response_format":
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "relation_extraction", "schema": RELATION_SCHEMA, "strict": True},
        }
    elif structured_mode == "guided_json":
        kwargs["extra_body"] = {"guided_json": RELATION_SCHEMA}
    return client.chat.completions.create(**kwargs)


def call_llm(client, model, text, temperature=0.0, max_retries=3, timeout=60, debug=False):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(text)},
    ]

    strategies = ["response_format", "guided_json", "none"]
    last_error = None

    for strategy in strategies:
        for attempt in range(max_retries):
            try:
                resp = _one_api_call(client, model, messages, temperature, 400, timeout, strategy)

                if debug:
                    print(f"  [debug] strategy={strategy} attempt={attempt} raw_response={resp}")

                choice = resp.choices[0]
                text_out, source = _extract_text_from_message(choice.message)

                if text_out is None:
                    print(f"  [warn] strategy={strategy} attempt={attempt+1}: empty content AND "
                          f"empty reasoning_content. finish_reason={choice.finish_reason}")
                    last_error = "empty_response"
                    time.sleep(1.0 * (attempt + 1))
                    continue

                cleaned = _strip_code_fences(text_out)
                try:
                    parsed = json.loads(cleaned)
                except json.JSONDecodeError as je:
                    print(f"  [warn] strategy={strategy} attempt={attempt+1}: JSON parse failed "
                          f"(source={source}): {je}. Raw snippet: {text_out[:200]!r}")
                    last_error = f"json_decode_error: {je}"
                    time.sleep(1.0 * (attempt + 1))
                    continue

                if parsed.get("relation") in RELATION_LABELS and isinstance(parsed.get("evidence_spans"), list):
                    if attempt > 0 or strategy != "response_format":
                        print(f"  [info] succeeded via strategy={strategy} on attempt={attempt+1}")
                    return parsed
                else:
                    print(f"  [warn] strategy={strategy} attempt={attempt+1}: parsed JSON missing/invalid "
                          f"required fields: {parsed}")
                    last_error = "schema_mismatch"
                    time.sleep(1.0 * (attempt + 1))

            except Exception as e:
                err_str = str(e)
                print(f"  [warn] strategy={strategy} attempt={attempt+1}: API call raised: {err_str[:300]}")
                last_error = err_str
                if any(k in err_str.lower() for k in ["response_format", "guided_json", "not supported",
                                                         "unrecognized", "400"]):
                    break
                time.sleep(1.0 * (attempt + 1))

    print(f"  [error] all structured-output strategies exhausted. last_error={last_error}")
    return None


# ----------------------------------------------------------------------------
# 2. Provenance gate (post-hoc validation of self-reported evidence)
# ----------------------------------------------------------------------------

def locate_spans_fuzzy(text, evidence_spans, min_span_chars=8, fuzzy_threshold=0.85):
    """Exact substring match first; if that fails, fall back to the longest
    matching substring (difflib) and accept it if it covers at least
    fuzzy_threshold of the cited span's length. This prevents a merely
    paraphrased (not fabricated) citation from being scored as zero
    coverage."""
    ranges = []
    for span in evidence_spans:
        span = span.strip()
        if len(span) < min_span_chars:
            continue
        idx = text.find(span)
        if idx != -1:
            ranges.append((idx, idx + len(span)))
            continue
        matcher = difflib.SequenceMatcher(None, text, span, autojunk=False)
        match = matcher.find_longest_match(0, len(text), 0, len(span))
        if match.size >= fuzzy_threshold * len(span) and match.size >= min_span_chars:
            ranges.append((match.a, match.a + match.size))
    coverage_chars = sum(e - s for s, e in ranges)
    return ranges, coverage_chars


def mask_ranges(text, ranges, placeholder="[REDACTED]"):
    if not ranges:
        return text
    ranges_sorted = sorted(ranges, key=lambda r: -r[0])
    out = text
    for s, e in ranges_sorted:
        out = out[:s] + placeholder + out[e:]
    return out


def gated_llm_call(client, model, text, min_coverage_chars=10, min_coverage_frac=0.03,
                    necessity_drop_required=True, necessity_n_probes=3,
                    necessity_agreement_threshold=0.5, fuzzy_threshold=0.85,
                    temperature=0.0, debug=False):
    """Coverage check uses locate_spans_fuzzy(); necessity check runs
    necessity_n_probes independent masked re-queries and requires at least
    necessity_agreement_threshold of them to disagree with the original
    prediction before crediting necessity as satisfied -- a single unlucky
    re-query can no longer flip the whole abstain/answer decision."""
    parsed = call_llm(client, model, text, temperature=temperature, debug=debug)
    if parsed is None:
        return {"abstain": True, "reason": "llm_call_failed", "relation": None,
                "evidence_spans": [], "evidence_ranges": [], "coverage_frac": 0.0}

    ranges, coverage_chars = locate_spans_fuzzy(text, parsed["evidence_spans"],
                                                 fuzzy_threshold=fuzzy_threshold)
    coverage_frac = coverage_chars / max(1, len(text))

    if not ranges or coverage_chars < min_coverage_chars or coverage_frac < min_coverage_frac:
        return {"abstain": True, "reason": "low_coverage", "relation": parsed["relation"],
                "evidence_spans": parsed["evidence_spans"], "evidence_ranges": ranges,
                "coverage_frac": coverage_frac, "confidence": parsed.get("confidence")}

    result = {"abstain": False, "relation": parsed["relation"], "evidence_spans": parsed["evidence_spans"],
               "evidence_ranges": ranges, "coverage_frac": coverage_frac,
               "confidence": parsed.get("confidence")}

    if necessity_drop_required:
        masked_text = mask_ranges(text, ranges)
        changed_count = 0
        n_valid_probes = 0
        probe_relations = []
        for _ in range(necessity_n_probes):
            parsed_masked = call_llm(client, model, masked_text, temperature=temperature, debug=debug)
            if parsed_masked is not None:
                n_valid_probes += 1
                probe_relations.append(parsed_masked["relation"])
                if parsed_masked["relation"] != parsed["relation"]:
                    changed_count += 1

        if n_valid_probes == 0:
            result["abstain"] = True
            result["reason"] = "necessity_probe_failed"
        else:
            necessity_satisfied = (changed_count / n_valid_probes) >= necessity_agreement_threshold
            result["necessity_satisfied"] = necessity_satisfied
            result["necessity_probe_relations"] = probe_relations
            result["necessity_change_fraction"] = changed_count / n_valid_probes
            if not necessity_satisfied:
                result["abstain"] = True
                result["reason"] = "necessity_check_failed"

    return result


def baseline_llm_call(client, model, text, temperature=0.0, debug=False):
    parsed = call_llm(client, model, text, temperature=temperature, debug=debug)
    if parsed is None:
        return {"abstain": True, "reason": "llm_call_failed", "relation": None,
                "evidence_spans": [], "evidence_ranges": [], "coverage_frac": 0.0}
    ranges, coverage_chars = locate_spans_fuzzy(text, parsed["evidence_spans"], min_span_chars=1)
    return {"abstain": False, "relation": parsed["relation"], "evidence_spans": parsed["evidence_spans"],
            "evidence_ranges": ranges, "coverage_frac": coverage_chars / max(1, len(text)),
            "confidence": parsed.get("confidence")}


# ----------------------------------------------------------------------------
# 2b. Standalone diagnostic pass for specific note_ids
# ----------------------------------------------------------------------------

def diagnose_notes(client, model, df_all, note_ids, pf_gate_call, temperature=0.0):
    print(f"\n[diagnose] running gated_llm_call() with debug=True on {len(note_ids)} note(s)...")
    for note_id in note_ids:
        matches = df_all[df_all["note_id"] == note_id]
        if matches.empty:
            print(f"  [diagnose] note_id={note_id!r} not found in generated dataset -- skipping.")
            continue
        row = matches.iloc[0]
        print(f"\n  --- {note_id} ---")
        print(f"  text: {row['text']}")
        out = pf_gate_call(client, model, row["text"], temperature=temperature, debug=True)
        print(f"  [diagnose] gated_llm_call output: "
              f"abstain={out['abstain']} reason={out.get('reason')} "
              f"coverage_frac={out.get('coverage_frac')} "
              f"necessity_satisfied={out.get('necessity_satisfied')} "
              f"necessity_change_fraction={out.get('necessity_change_fraction')}")


# ----------------------------------------------------------------------------
# 3. Evaluation
# ----------------------------------------------------------------------------

def char_ranges_to_recall(gold_span_start, gold_span_end, pred_ranges):
    if gold_span_start is None or gold_span_start == -1 or gold_span_end <= gold_span_start:
        return None
    covered = np.zeros(gold_span_end - gold_span_start, dtype=bool)
    for s, e in pred_ranges:
        os_, oe = max(s, gold_span_start), min(e, gold_span_end)
        if oe > os_:
            covered[os_ - gold_span_start: oe - gold_span_start] = True
    return covered.mean()


def evaluate_llm(call_fn, client, model, df, temperature=0.0, verbose_every=10, debug_first=False):
    correct, total, abstained = 0, 0, 0
    overlaps = []
    records = []
    for i, (_, row) in enumerate(df.iterrows()):
        out = call_fn(client, model, row["text"], temperature=temperature,
                       debug=(debug_first and i == 0))
        total += 1
        gold_span = row.get("gold_evidence_span")
        gold_start = row["text"].find(gold_span) if isinstance(gold_span, str) else -1
        gold_end = gold_start + len(gold_span) if gold_start != -1 else -1

        if out["abstain"]:
            abstained += 1
        else:
            if out["relation"] == row["gold_relation"]:
                correct += 1
            recall = char_ranges_to_recall(gold_start, gold_end, out["evidence_ranges"])
            if recall is not None:
                overlaps.append(recall)

        rec = dict(out)
        rec["note_id"] = row["note_id"]
        rec["gold_relation"] = row["gold_relation"]
        records.append(rec)

        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"    processed {i+1}/{len(df)} (abstained so far: {abstained})")

    return {
        "accuracy_on_answered": correct / max(1, total - abstained),
        "abstain_rate": abstained / total,
        "mean_evidence_recall": float(np.mean(overlaps)) if overlaps else 0.0,
        "n": total,
    }, records


# ----------------------------------------------------------------------------
# 3b. Axis-3 integration
# ----------------------------------------------------------------------------

def _is_real_disagreement(rel_base, rel_pf):
    if rel_base is None and rel_pf is None:
        return False
    if (rel_base is None) != (rel_pf is None):
        return True
    return rel_base != rel_pf


def build_knowledge_graph_llm(client, model, df, gate_fn_baseline, gate_fn_pf,
                               site_id, temperature=0.0, verbose_every=10):
    kg = ProvenanceTemporalKG(site_id=site_id)
    n_processed = 0
    n_disagreements = 0

    for i, (_, row) in enumerate(df.iterrows()):
        baseline_out = gate_fn_baseline(client, model, row["text"], temperature=temperature)
        pf_out = gate_fn_pf(client, model, row["text"], temperature=temperature)

        _, _, ek_base, rel_base = ingest_llm_result(
            kg, note_id=row["note_id"], event1_label=row["event1"], event2_label=row["event2"],
            llm_result=baseline_out, model_id=f"{model}_baseline")
        e1, e2, ek_pf, rel_pf = ingest_llm_result(
            kg, note_id=row["note_id"], event1_label=row["event1"], event2_label=row["event2"],
            llm_result=pf_out, model_id=f"{model}_provenance_first")

        if _is_real_disagreement(rel_base, rel_pf):
            n_disagreements += 1

        n_processed += 1
        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"    [kg] processed {i+1}/{len(df)} (disagreements so far: {n_disagreements})")

    print(f"  [kg] ingested {n_processed} notes x 2 models into site '{site_id}' "
          f"({kg.graph.number_of_nodes()} nodes, {kg.graph.number_of_edges()} edges, "
          f"{n_disagreements} baseline/provenance-first disagreements detected and cross-flagged)")
    return kg, n_disagreements


def report_contradictions(kg, max_examples=5):
    seen_pairs = set()
    examples = []
    for u, v, k, d in kg.graph.edges(keys=True, data=True):
        if (u, v) in seen_pairs:
            continue
        seen_pairs.add((u, v))
        active = kg.get_active_relations(u, v)
        distinct_relation_types = {edge_data["relation_type"] for _, edge_data in active}
        if len(active) > 1 and len(distinct_relation_types) > 1:
            u_label = kg.graph.nodes[u].get("label", u)
            v_label = kg.graph.nodes[v].get("label", v)
            examples.append({
                "event1": u_label, "event2": v_label,
                "relations": [(edge_data["relation_type"], edge_data["model_id"],
                                edge_data["provenance"].get("coverage"))
                               for _, edge_data in active],
            })
    print(f"\n  [kg] {len(examples)} event pairs have >=2 active relations with DISTINCT "
          f"relation_types (i.e. genuine contradictions, not just repeated agreement) "
          f"(sample of up to {max_examples}):")
    for ex in examples[:max_examples]:
        print(f"    event1={ex['event1']!r} event2={ex['event2']!r}")
        for rel_type, model_id, coverage in ex["relations"]:
            print(f"      -> {rel_type} (model={model_id}, coverage={coverage})")
    return examples


# ----------------------------------------------------------------------------
# 4. Provenance manipulation experiment
# ----------------------------------------------------------------------------

def get_fabricated_char_span(text):
    """Legacy fallback pattern-match, used only when a row has no
    injected_spans column (see get_injected_char_spans below) -- i.e. only
    for CSVs from an older/foreign generator. Not exercised by
    synthetic_longitudinal_generator_v1.py output, which always populates
    injected_spans."""
    m = re.search(r"Notably,.*?not treated\.", text)
    return (m.start(), m.end()) if m else None


def resolve_base_note_id(row):
    """Returns the note_id of the un-perturbed base note this adversarial
    row derives from. Uses synthetic_longitudinal_generator_v1.py's
    explicit base_note_id column when present; falls back to a legacy
    _adv/_nat suffix-stripping heuristic otherwise."""
    explicit = row.get("base_note_id")
    if isinstance(explicit, str) and explicit:
        return explicit
    return row["note_id"].replace("_adv", "_base").replace("_nat", "_base")


def get_injected_char_spans(row):
    """Returns a list of (start, end) char spans for injected/adversarial
    content in this row's text. Prefers the explicit injected_spans JSON
    column written by synthetic_longitudinal_generator_v1.py (supports
    multiple, typed contradiction spans: fabricated_symptom,
    temporal_contradiction, dosage_contradiction, negation_flip). Falls
    back to get_fabricated_char_span() for rows without that column."""
    raw = row.get("injected_spans")
    if isinstance(raw, str) and raw.strip() and raw.strip() != "[]":
        try:
            spans = json.loads(raw)
            return [(s["start"], s["end"]) for s in spans]
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    span = get_fabricated_char_span(row["text"])
    return [span] if span else []


def run_manipulation_eval_llm(call_fn, client, model, df_manip, all_base_lookup, temperature=0.0):
    res = {"adversarial": {"n": 0, "abstain": 0, "pred_stable": 0, "evidence_shift": 0},
           "natural_variability": {"n": 0, "abstain": 0, "pred_stable": 0, "evidence_recall": []}}

    for _, row in df_manip.iterrows():
        out = call_fn(client, model, row["text"], temperature=temperature)
        base_id = resolve_base_note_id(row)
        base_row = all_base_lookup.get(base_id)
        if base_row is None:
            continue
        base_out = call_fn(client, model, base_row["text"], temperature=temperature)

        if row["variant"] == "adversarial":
            d = res["adversarial"]
            d["n"] += 1
            if out["abstain"]:
                d["abstain"] += 1
                continue
            if not base_out["abstain"] and out["relation"] == base_out["relation"]:
                d["pred_stable"] += 1
            char_spans = get_injected_char_spans(row)
            shifted = any(
                any(min(e, fe) > max(s, fs) for s, e in out["evidence_ranges"])
                for (fs, fe) in char_spans
            )
            if shifted:
                d["evidence_shift"] += 1
        else:
            d = res["natural_variability"]
            d["n"] += 1
            if out["abstain"]:
                d["abstain"] += 1
                continue
            if not base_out["abstain"] and out["relation"] == base_out["relation"]:
                d["pred_stable"] += 1
            gold_span = row.get("gold_evidence_span")
            gold_start = row["text"].find(gold_span) if isinstance(gold_span, str) else -1
            gold_end = gold_start + len(gold_span) if gold_start != -1 else -1
            recall = char_ranges_to_recall(gold_start, gold_end, out["evidence_ranges"])
            if recall is not None:
                d["evidence_recall"].append(recall)
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
# 5. Dataset loading (synthetic_longitudinal_generator_v1.py output ONLY --
#    this is now the sole source of data for this script)
# ----------------------------------------------------------------------------

def load_external_dataset(data_dir, test_frac=0.15, dev_frac=0.15, seed=7):
    """Loads notes/adversarial CSVs produced by
    synthetic_longitudinal_generator_v1.py and reshapes them into
    train/dev/test/manip dataframes.

    Splitting logic is intentionally IDENTICAL to
    provenance_first_poc_clinicalBERT_v6.py's function of the same name:
    same seed-based patient shuffle, same test_frac/dev_frac defaults.
    Given the same --data_dir and --seed, this guarantees a v6 run and a
    v7 run evaluate on the identical held-out patient split -- required
    for compare_bert_llm_results.py's comparison table to be a fair
    like-for-like comparison rather than an artifact of different samples.
    """
    notes_path = os.path.join(data_dir, "synthetic_notes_v1.csv")
    adv_path = os.path.join(data_dir, "synthetic_adversarial_v1.csv")
    if not os.path.exists(notes_path):
        raise FileNotFoundError(
            f"{notes_path} not found. Run synthetic_longitudinal_generator_v1.py "
            f"--out_dir {data_dir} once first (do not regenerate before every run -- "
            f"see module docstring).")

    df_all_notes = pd.read_csv(notes_path)
    n_notes_total = len(df_all_notes)

    df_rel = df_all_notes[df_all_notes["gold_relation"].notna()].copy()
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
        df_manip = df_adv
    else:
        print(f"  [warn] {adv_path} not found -- manipulation-eval dataframe will be empty.")

    print(f"  loaded {n_notes_total} total notes ({len(df_rel)} usable for relation "
          f"classification: train={len(df_train)} dev={len(df_dev)} test={len(df_test)}, "
          f"split by patient_id, seed={seed}) manip_pairs={len(df_manip)}")
    return df_train, df_dev, df_test, df_manip


# ----------------------------------------------------------------------------
# 6. Dual robustness / provenance-stability certification (CERTIFY) [AXIS 2]
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
        w = words[i]; core = w.strip(".,").lower(); suffix = w[len(core):]
        words[i] = ABBREV_MAP.get(core, core) + suffix
    return " ".join(words)


def _perturb_ocr_noise(text, k):
    chars = list(text)
    positions = [i for i, c in enumerate(chars) if c.lower() in OCR_CONFUSABLES]
    random.shuffle(positions)
    for p in positions[:k]:
        c = chars[p]; repl = OCR_CONFUSABLES[c.lower()]
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
        w = words[i]; core = w.strip(".,").lower(); suffix = w[len(core):]
        words[i] = SYNONYM_MAP.get(core, core) + suffix
    return " ".join(words)


_PERTURB_FUNCS = {
    "abbreviation_swap": _perturb_abbreviation_swap, "ocr_noise": _perturb_ocr_noise,
    "token_insertion": _perturb_token_insertion, "synonym_substitution": _perturb_synonym_substitution,
}


def sample_perturbation(text, perturbation_space, epsilon_budget):
    ptype = random.choice(perturbation_space)
    k = random.randint(1, epsilon_budget)
    return _PERTURB_FUNCS[ptype](text, k), ptype


def char_range_jaccard(ranges_a, ranges_b, text_len):
    def to_bitset(ranges):
        s = set()
        for a, b in ranges:
            s.update(range(max(0, a), min(text_len, b)))
        return s
    a, b = to_bitset(ranges_a), to_bitset(ranges_b)
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def clopper_pearson_lower(successes, n, alpha):
    if n == 0 or successes == 0:
        return 0.0
    return float(stats.beta.ppf(alpha, successes, n - successes + 1))


def certify_llm(client, model, text, gate_fn, perturbation_space=PERTURBATION_SPACE,
                 epsilon_budget=2, n_samples=100, alpha=0.05, theta_stability=0.5,
                 temperature=0.0):
    base_out = gate_fn(client, model, text, temperature=temperature)
    if base_out["abstain"]:
        return {"status": "UNCERTIFIED", "reason": "base_input_abstained",
                "prediction_confidence": 0.0, "provenance_confidence": 0.0,
                "p_pred_lower": 0.0, "p_evi_lower": 0.0}

    y0 = base_out["relation"]
    E0 = base_out["evidence_ranges"]

    pred_matches, evidence_matches = [], []
    perturbation_log = []

    for _ in range(n_samples):
        x_pert, ptype = sample_perturbation(text, perturbation_space, epsilon_budget)
        out = gate_fn(client, model, x_pert, temperature=temperature)
        perturbation_log.append(ptype)

        if out["abstain"]:
            pred_matches.append(0)
            evidence_matches.append(0)
            continue

        pred_matches.append(int(out["relation"] == y0))
        sim = char_range_jaccard(out["evidence_ranges"], E0, len(x_pert))
        evidence_matches.append(int(sim >= theta_stability))

    n_pred_success = sum(pred_matches)
    n_evi_success = sum(evidence_matches)
    p_pred_lower = clopper_pearson_lower(n_pred_success, n_samples, alpha / 2)
    p_evi_lower = clopper_pearson_lower(n_evi_success, n_samples, alpha / 2)
    certified = (p_pred_lower >= 1 - alpha) and (p_evi_lower >= 1 - alpha)

    result = {
        "status": "CERTIFIED" if certified else "UNCERTIFIED",
        "prediction_confidence": p_pred_lower, "provenance_confidence": p_evi_lower,
        "p_pred_lower": p_pred_lower, "p_evi_lower": p_evi_lower,
        "n_pred_success": n_pred_success, "n_evi_success": n_evi_success,
        "n_samples": n_samples, "alpha": alpha, "theta_stability": theta_stability,
        "epsilon_budget": epsilon_budget, "perturbation_space": perturbation_space,
        "perturbation_counts": dict(Counter(perturbation_log)),
    }
    if not certified:
        result["reason"] = "prediction" if p_pred_lower < 1 - alpha else "provenance"
    return result


def run_certification_suite_llm(client, model, df_test, gate_fn, n_examples=20, n_samples=100,
                                  alpha=0.05, epsilon_budget=2, theta_stability=0.5):
    sample_df = df_test.sample(n=min(n_examples, len(df_test)), random_state=0)
    certs = []
    for _, row in sample_df.iterrows():
        cert = certify_llm(client, model, row["text"], gate_fn, n_samples=n_samples, alpha=alpha,
                             epsilon_budget=epsilon_budget, theta_stability=theta_stability)
        cert["note_id"] = row["note_id"]
        certs.append(cert)

    n_certified = sum(1 for c in certs if c["status"] == "CERTIFIED")
    n_base_abstained = sum(1 for c in certs if c.get("reason") == "base_input_abstained")
    summary = {
        "n_examples": len(certs), "n_certified": n_certified,
        "n_base_abstained": n_base_abstained,
        "certification_rate": n_certified / len(certs) if certs else 0.0,
        "mean_prediction_confidence": float(np.mean([c["prediction_confidence"] for c in certs])),
        "mean_provenance_confidence": float(np.mean([c["provenance_confidence"] for c in certs])),
        "per_example": certs,
    }
    if n_base_abstained > 0:
        print(f"  [warn] {n_base_abstained}/{len(certs)} certification examples abstained on the "
              f"UNPERTURBED base input -- the gate is not producing predictions on clean data for "
              f"these examples, which caps certification_rate regardless of robustness to "
              f"perturbation. Use --diagnose_notes to inspect these specific note_ids before "
              f"trusting this result; consider loosening --min_coverage_chars / "
              f"--min_coverage_frac or raising --necessity_n_probes.")
    required_ceiling = clopper_pearson_lower(n_samples, n_samples, alpha / 2)
    if required_ceiling < 1 - alpha:
        print(f"  [info] n_samples={n_samples} at alpha={alpha}: the Clopper-Pearson lower bound "
              f"cannot exceed ~{required_ceiling:.3f} even with a perfect success rate, which is "
              f"below the 1-alpha={1-alpha:.2f} threshold required for CERTIFIED status. Raise "
              f"--cert_n_samples if you want a realistic chance of positive certificates.")
    else:
        print(f"  [info] n_samples={n_samples} at alpha={alpha}: a perfect success rate CAN reach "
              f"CERTIFIED status (ceiling={required_ceiling:.3f} >= {1-alpha:.2f}). Any UNCERTIFIED "
              f"result at this sample size reflects actual measured instability, not a statistical "
              f"sample-size ceiling.")
    return summary


# ----------------------------------------------------------------------------
# 7. Main
# ----------------------------------------------------------------------------

def sample_manipulation_pairs(df_manip, n_per_variant, seed):
    parts = []
    for _, group in df_manip.groupby("variant"):
        n = min(n_per_variant, len(group))
        parts.append(group.sample(n=n, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_base", default="http://gpusoif:8000/v1")
    parser.add_argument("--api_key", default=os.getenv("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--llm_model", default="mistralai/Mistral-Small-3.2-24B-Instruct-2506")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out_dir", default="output")
    parser.add_argument("--n_test", type=int, default=60)
    parser.add_argument("--n_manip_pairs", type=int, default=60)

    parser.add_argument("--min_coverage_chars", type=int, default=10)
    parser.add_argument("--min_coverage_frac", type=float, default=0.03)

    parser.add_argument("--necessity_n_probes", type=int, default=3)
    parser.add_argument("--necessity_agreement_threshold", type=float, default=0.5)
    parser.add_argument("--fuzzy_span_threshold", type=float, default=0.85)

    parser.add_argument("--run_certification", action="store_true")
    parser.add_argument("--cert_n_examples", type=int, default=20)
    parser.add_argument("--cert_n_samples", type=int, default=100)
    parser.add_argument("--cert_alpha", type=float, default=0.05)
    parser.add_argument("--cert_epsilon_budget", type=int, default=2)
    parser.add_argument("--cert_theta_stability", type=float, default=0.5)

    parser.add_argument("--debug_first_call", action="store_true")
    parser.add_argument("--diagnose_notes", nargs="*", default=[])
    parser.add_argument("--site_id", default="llm_site")
    parser.add_argument("--skip_kg", action="store_true")
    parser.add_argument("--kg_n_notes", type=int, default=None)

    parser.add_argument("--data_dir", required=True,
                         help="REQUIRED. Directory already populated by "
                              "synthetic_longitudinal_generator_v1.py (must contain "
                              "synthetic_notes_v1.csv and, optionally, "
                              "synthetic_adversarial_v1.csv). This script no longer generates "
                              "its own synthetic data -- run the generator once, separately, "
                              "and point every v6/v7 run at the same directory and --seed so "
                              "they share an identical held-out patient split.")
    args = parser.parse_args()

    print(f"Connecting to LLM API at {args.api_base}, model={args.llm_model}")
    client = OpenAI(base_url=args.api_base, api_key=args.api_key)

    if args.debug_first_call:
        print("[debug] Issuing a single raw diagnostic call before starting the pipeline...")
        probe = call_llm(client, args.llm_model,
                          "On day 1, the patient was diagnosed with pneumonia. "
                          "Antibiotics were initiated on day 2.",
                          temperature=args.temperature, debug=True)
        print(f"[debug] Diagnostic call result: {probe}\n")

    print(f"[1/7] Loading dataset from {args.data_dir} "
          f"(synthetic_longitudinal_generator_v1.py output)...")
    df_train, df_dev, df_test, df_manip = load_external_dataset(args.data_dir, seed=args.seed)
    print(f"  train={len(df_train)} dev={len(df_dev)} test={len(df_test)} manip_pairs={len(df_manip)}")

    all_base_lookup = {r["note_id"]: r for _, r in pd.concat([df_train, df_dev, df_test]).iterrows()}
    df_all = pd.concat([df_train, df_dev, df_test], ignore_index=True)

    def pf_gate_call(client_, model_, text_, temperature=0.0, debug=False):
        return gated_llm_call(client_, model_, text_, min_coverage_chars=args.min_coverage_chars,
                               min_coverage_frac=args.min_coverage_frac,
                               necessity_n_probes=args.necessity_n_probes,
                               necessity_agreement_threshold=args.necessity_agreement_threshold,
                               fuzzy_threshold=args.fuzzy_span_threshold,
                               temperature=temperature, debug=debug)

    if args.diagnose_notes:
        diagnose_notes(client, args.llm_model, df_all, args.diagnose_notes, pf_gate_call,
                        temperature=args.temperature)

    test_sample = df_test.sample(n=min(args.n_test, len(df_test)), random_state=args.seed)
    manip_sample = (sample_manipulation_pairs(df_manip, args.n_manip_pairs, args.seed)
                     if len(df_manip) else df_manip)
    if len(manip_sample):
        assert "variant" in manip_sample.columns, "manip_sample lost its 'variant' column -- sampling bug"

    print(f"[2/7] Evaluating BASELINE (no gate) on {len(test_sample)} test notes via LLM API...")
    baseline_metrics, baseline_records = evaluate_llm(baseline_llm_call, client, args.llm_model,
                                                        test_sample, temperature=args.temperature)
    print(f"  baseline: {baseline_metrics}")

    print(f"[3/7] Evaluating PROVENANCE-FIRST (fuzzy coverage + multi-probe necessity gate) on "
          f"{len(test_sample)} test notes via LLM API...")
    pf_metrics, pf_records = evaluate_llm(pf_gate_call, client, args.llm_model,
                                            test_sample, temperature=args.temperature)
    print(f"  provenance_first: {pf_metrics}")
    if pf_metrics["abstain_rate"] > 0.5:
        print(f"  [warn] provenance_first abstains on {pf_metrics['abstain_rate']:.1%} of clean test "
              f"notes. Consider loosening --min_coverage_chars / --min_coverage_frac, or check "
              f"--necessity_agreement_threshold, before trusting downstream robustness numbers.")

    results = {
        "config": vars(args),
        "test_metrics": {"baseline": baseline_metrics, "provenance_first": pf_metrics},
    }

    if len(manip_sample):
        print(f"[4/7] Running provenance-manipulation experiment on {len(manip_sample)} paired notes...")
        results["manipulation_metrics"] = {
            "baseline": summarize_manipulation(
                run_manipulation_eval_llm(baseline_llm_call, client, args.llm_model, manip_sample,
                                            all_base_lookup, temperature=args.temperature)),
            "provenance_first": summarize_manipulation(
                run_manipulation_eval_llm(pf_gate_call, client, args.llm_model, manip_sample,
                                            all_base_lookup, temperature=args.temperature)),
        }
        print(f"  baseline manip: {results['manipulation_metrics']['baseline']}")
        print(f"  provenance_first manip: {results['manipulation_metrics']['provenance_first']}")

        base_adv_stab = results["manipulation_metrics"]["baseline"]["adversarial_pred_stability"]
        pf_adv_stab = results["manipulation_metrics"]["provenance_first"]["adversarial_pred_stability"]
        if pf_adv_stab < base_adv_stab:
            print(f"  [warn] provenance_first adversarial prediction stability ({pf_adv_stab:.3f}) is "
                  f"LOWER than baseline ({base_adv_stab:.3f}). If --necessity_n_probes > 1 did not "
                  f"close most of this gap versus a previous run, the instability is likely coming "
                  f"from LLM sampling noise rather than genuine content-sensitivity.")
    else:
        print("[4/7] No manipulation pairs available -- skipping manipulation experiment.")

    if args.run_certification:
        print(f"[5/7] Running dual robustness / provenance-stability certification "
              f"(n_examples={args.cert_n_examples}, n_samples={args.cert_n_samples})...")
        results["certification"] = {
            "baseline": run_certification_suite_llm(
                client, args.llm_model, df_test, baseline_llm_call, n_examples=args.cert_n_examples,
                n_samples=args.cert_n_samples, alpha=args.cert_alpha,
                epsilon_budget=args.cert_epsilon_budget, theta_stability=args.cert_theta_stability),
            "provenance_first": run_certification_suite_llm(
                client, args.llm_model, df_test, pf_gate_call, n_examples=args.cert_n_examples,
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
        print("[5/7] Skipping certification (pass --run_certification to enable).")

    if not args.skip_kg:
        kg_n_notes = args.kg_n_notes or args.n_test
        kg_sample = df_test.sample(n=min(kg_n_notes, len(df_test)), random_state=args.seed)
        print(f"[6/7] Building shared provenance temporal knowledge graph from baseline + "
              f"provenance-first on {len(kg_sample)} notes (site_id={args.site_id})...")
        kg, n_disagreements = build_knowledge_graph_llm(
            client, args.llm_model, kg_sample, baseline_llm_call, pf_gate_call,
            site_id=args.site_id, temperature=args.temperature)

        contradiction_examples = report_contradictions(kg)

        kg_summary = aggregate_temporal_motifs(kg)
        print(f"  [kg] site summary: {kg_summary}")
        results["knowledge_graph_summary"] = {
            "site_summary": kg_summary, "n_disagreements": n_disagreements,
            "n_contradiction_pairs": len(contradiction_examples),
            "contradiction_examples": contradiction_examples,
        }

        kg_path = os.path.join(args.out_dir, "provenance_kg_llm.pkl")
        with open(kg_path, "wb") as f:
            pickle.dump(kg, f)
        print(f"  [kg] graph pickled to {kg_path}")
    else:
        print("[6/7] Skipping knowledge graph construction (--skip_kg set).")

    out_path = os.path.join(args.out_dir, "poc_results_llm_v7.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[7/7] Done. Results written to {out_path}")


if __name__ == "__main__":
    main()
