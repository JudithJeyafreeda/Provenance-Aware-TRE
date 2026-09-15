"""
RO2 Compression Operators: Aggregation, Revision-Representation, Temporal-Approximation
===========================================================================================


Usage
-----
    python RO2_compression_operators.py --data_dir "path/to/data" --confidence_threshold 0.65 --fixed_window_k 2 --n_query_samples_per_patient 6 --seed 11
"""

import os
import json
import random
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# 1. Load generator output
# ----------------------------------------------------------------------------

def load_ro2_inputs(data_dir):
    notes_path = os.path.join(data_dir, "synthetic_notes_v1.csv")
    facts_path = os.path.join(data_dir, "synthetic_facts_v1.csv")
    edges_path = os.path.join(data_dir, "synthetic_revision_edges_v1.csv")
    for p in (notes_path, facts_path, edges_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found -- run synthetic_longitudinal_generator_v1.py first.")

    notes = pd.read_csv(notes_path)
    facts = pd.read_csv(facts_path)
    edges = pd.read_csv(edges_path)

    if "mentioned_fact_ids" not in notes.columns:
        raise ValueError(
            "synthetic_notes_v1.csv has no mentioned_fact_ids column -- regenerate with "
            "the current synthetic_longitudinal_generator_v1.py (mention linkage is "
            "required to build the Evidence Graph; older CSVs predate this column).")
    notes["mentioned_fact_ids"] = notes["mentioned_fact_ids"].apply(
        lambda s: json.loads(s) if isinstance(s, str) and s.strip() else [])
    return facts, edges, notes


# ----------------------------------------------------------------------------
# 2. Ground-truth oracle (same logic as the generator's
#    evaluate_revision_sensitive_state -- duplicated here rather than
#    imported so this script has no hard dependency on the generator
#    module, only on its CSV output)
# ----------------------------------------------------------------------------

def evaluate_revision_sensitive_state(facts, query_day):
    by_concept = {}
    for f in facts:
        if f["day"] > query_day:
            continue
        superseded = any(g["revises"] == f["fact_id"] and g["day"] <= query_day for g in facts
                          if pd.notna(g["revises"]))
        if not superseded:
            by_concept[f["concept_type"]] = f["fact_id"]
    return by_concept


# ----------------------------------------------------------------------------
# 3. Simulated per-mention extraction confidence
# ----------------------------------------------------------------------------

STYLE_BASE_CONFIDENCE = {
    "verbose_discharge": 0.85, "structured_admission": 0.75,
    "nursing_note": 0.60, "terse_icu": 0.55,
}


def simulate_mention_confidence(style, rng):
    base = STYLE_BASE_CONFIDENCE.get(style, 0.7)
    return float(np.clip(rng.gauss(base, 0.08), 0.05, 0.99))


# ----------------------------------------------------------------------------
# 4. Uncompressed Evidence Graph
# ----------------------------------------------------------------------------

def make_uncompressed_representation(facts, edges, notes, rng):
    facts_by_id = {f["fact_id"]: f for _, f in facts.iterrows()}
    nodes, edge_list = [], []

    for _, note in notes.iterrows():
        for fact_id in note["mentioned_fact_ids"]:
            if fact_id not in facts_by_id:
                continue
            f = facts_by_id[fact_id]
            conf = simulate_mention_confidence(note["style"], rng)
            mid = f"{note['note_id']}__{fact_id}"
            nodes.append({
                "id": mid, "kind": "mention", "fact_id": fact_id,
                "patient_id": f["patient_id"], "concept_type": f["concept_type"],
                "value": f["value"], "day": int(note["day"]), "confidence": conf,
                "note_id": note["note_id"],
            })
            edge_list.append({"type": "derived_from", "from": mid, "to": note["note_id"]})

    for _, r in edges.iterrows():
        # "to_fact_id revises from_fact_id" -- edge direction: new -> old
        edge_list.append({"type": "revises", "from": r["to_fact_id"], "to": r["from_fact_id"],
                           "day": int(r["day"])})

    return {"nodes": nodes, "edges": edge_list}


def representation_size(rep):
    return {"n_nodes": len(rep["nodes"]), "n_edges": len(rep["edges"]),
            "total": len(rep["nodes"]) + len(rep["edges"])}


def provenance_recall(rep, uncompressed):
    all_mention_ids = {n["id"] for n in uncompressed["nodes"] if n["kind"] == "mention"}
    if not all_mention_ids:
        return 1.0
    reachable = {n["id"] for n in rep["nodes"] if n["kind"] == "mention"}
    for e in rep["edges"]:
        if e["type"] == "membership" and "orig_mention_id" in e:
            reachable.add(e["orig_mention_id"])
    return len(reachable & all_mention_ids) / len(all_mention_ids)


# ----------------------------------------------------------------------------
# 5. Operator 1: AGGREGATION
# ----------------------------------------------------------------------------

def apply_aggregation(rep):
    """Alignment rule: same fact_id (== same concept, same underlying
    event) merges into one state node. In this synthetic corpus mentions
    of the same fact_id never conflict with each other (contradictions
    only appear in the separate adversarial CSV), so the alignment rule
    reduces to 'group by fact_id'. Membership edges retain every original
    mention_id + note_id + confidence, so this is reversible by
    construction -- provenance_recall should come out at 1.0."""
    mention_nodes = [n for n in rep["nodes"] if n["kind"] == "mention"]
    other_nodes = [n for n in rep["nodes"] if n["kind"] != "mention"]
    by_fact = defaultdict(list)
    for n in mention_nodes:
        by_fact[n["fact_id"]].append(n)

    new_nodes = list(other_nodes)
    new_edges = [e for e in rep["edges"] if e["type"] != "derived_from"]
    for fact_id, mentions in by_fact.items():
        rep0 = mentions[0]
        state_id = f"state__{fact_id}"
        new_nodes.append({
            "id": state_id, "kind": "state", "fact_id": fact_id,
            "patient_id": rep0["patient_id"], "concept_type": rep0["concept_type"],
            "value": rep0["value"], "day": min(m["day"] for m in mentions),
            "confidence": max(m["confidence"] for m in mentions),
        })
        for m in mentions:
            new_edges.append({"type": "membership", "from": state_id, "to": m["note_id"],
                               "orig_mention_id": m["id"], "mention_day": m["day"],
                               "mention_confidence": m["confidence"]})
    return {"nodes": new_nodes, "edges": new_edges}


# ----------------------------------------------------------------------------
# 6. Operator 2: REVISION_REPRESENTATION
# ----------------------------------------------------------------------------

def apply_revision_representation(rep):
    """Operates on an aggregated (fact-level state-node) representation.
    Walks each revises chain (from newest fact back to its oldest
    ancestor) and collapses it into a single evidential-state node
    carrying an ordered revision_history, replacing N state nodes +
    (N-1) revises edges with 1 node. Membership edges are re-attached to
    the collapsed node so provenance_recall is unaffected. Non-revised
    state nodes pass through untouched -- this degrades gracefully to a
    no-op for patients with no revision in this corpus."""
    state_nodes = {n["id"]: n for n in rep["nodes"] if n["kind"] == "state"}
    other_nodes = [n for n in rep["nodes"] if n["kind"] != "state"]
    revises_edges = [e for e in rep["edges"] if e["type"] == "revises"]
    membership_edges = [e for e in rep["edges"] if e["type"] == "membership"]
    other_edges = [e for e in rep["edges"] if e["type"] not in ("revises",)]

    fact_id_to_state_id = {n["fact_id"]: n["id"] for n in state_nodes.values()}
    revises_by_new = {e["from"]: e["to"] for e in revises_edges}  # new_fact_id -> old_fact_id
    all_new, all_old = set(revises_by_new.keys()), set(revises_by_new.values())
    chain_heads = [fid for fid in fact_id_to_state_id if fid in all_new and fid not in all_old]

    collapsed_nodes, collapsed_state_ids_used, new_membership_edges = [], set(), []

    for head_fact_id in chain_heads:
        chain = [head_fact_id]
        cur = head_fact_id
        while cur in revises_by_new:
            cur = revises_by_new[cur]
            chain.append(cur)
        chain = list(reversed(chain))  # oldest -> newest

        history = []
        state_id = f"revstate__{'_'.join(chain)}"
        for fid in chain:
            sn = state_nodes[fact_id_to_state_id[fid]]
            history.append({"fact_id": fid, "value": sn["value"], "day": sn["day"],
                             "confidence": sn["confidence"]})
            collapsed_state_ids_used.add(fact_id_to_state_id[fid])
            for me in membership_edges:
                if me["from"] == fact_id_to_state_id[fid]:
                    new_membership_edges.append({**me, "from": state_id})

        rep_recent = history[-1]
        collapsed_nodes.append({
            "id": state_id, "kind": "revision_state", "fact_ids": chain,
            "patient_id": state_nodes[fact_id_to_state_id[chain[0]]]["patient_id"],
            "concept_type": state_nodes[fact_id_to_state_id[chain[0]]]["concept_type"],
            "value": rep_recent["value"], "day": rep_recent["day"],
            "confidence": rep_recent["confidence"], "revision_history": history,
        })

    passthrough_states = [n for n in state_nodes.values() if n["id"] not in collapsed_state_ids_used]
    passthrough_ids = {n["id"] for n in passthrough_states}
    passthrough_membership = [e for e in membership_edges if e["from"] in passthrough_ids]

    new_nodes = other_nodes + passthrough_states + collapsed_nodes
    new_edges = ([e for e in other_edges if e["type"] != "membership"]
                 + passthrough_membership + new_membership_edges)
    return {"nodes": new_nodes, "edges": new_edges}


# ----------------------------------------------------------------------------
# 7. Operator 3: TEMPORAL_APPROXIMATION
# ----------------------------------------------------------------------------

def _allen_relation(day, anchor_day):
    if day < anchor_day:
        return "BEFORE"
    elif day > anchor_day:
        return "AFTER"
    return "OVERLAP"


def apply_temporal_approximation(rep, confidence_threshold, rng):
    """For any node (or, for revision_state nodes, any history entry)
    whose confidence falls below confidence_threshold, replaces its exact
    day with a temporally_related edge to a same-(patient,concept) anchor
    day (the earliest day among that patient/concept's HIGH-confidence
    entries) instead of the timestamp itself. This is the one operator
    that is allowed to be lossy -- see query_active_value()'s handling of
    day=None entries for how that loss propagates into query answers."""
    anchor_day = {}
    for n in rep["nodes"]:
        key = (n.get("patient_id"), n.get("concept_type"))
        days = []
        if n["kind"] in ("state", "mention") and n.get("confidence", 1.0) >= confidence_threshold:
            days.append(n["day"])
        elif n["kind"] == "revision_state":
            days += [h["day"] for h in n["revision_history"] if h["confidence"] >= confidence_threshold]
        if days:
            anchor_day[key] = min(anchor_day.get(key, min(days)), min(days))

    new_nodes, new_edges = [], list(rep["edges"])
    for n in rep["nodes"]:
        key = (n.get("patient_id"), n.get("concept_type"))
        anchor = anchor_day.get(key)
        if n["kind"] in ("state", "mention"):
            if n.get("confidence", 1.0) < confidence_threshold and anchor is not None:
                relation = _allen_relation(n["day"], anchor)
                new_edges.append({"type": "temporally_related", "from": n["id"],
                                   "relation": relation, "anchor_day": anchor})
                n = {**n, "day": None, "approx_relation": relation, "approx_anchor_day": anchor}
            new_nodes.append(n)
        elif n["kind"] == "revision_state":
            new_history = []
            for h in n["revision_history"]:
                if h["confidence"] < confidence_threshold and anchor is not None:
                    relation = _allen_relation(h["day"], anchor)
                    new_edges.append({"type": "temporally_related", "from": n["id"],
                                       "relation": relation, "anchor_day": anchor,
                                       "history_fact_id": h["fact_id"]})
                    h = {**h, "day": None, "approx_relation": relation, "approx_anchor_day": anchor}
                new_history.append(h)
            n = {**n, "revision_history": new_history, "day": new_history[-1]["day"]}
            new_nodes.append(n)
        else:
            new_nodes.append(n)
    return {"nodes": new_nodes, "edges": new_edges}


# ----------------------------------------------------------------------------
# 8. External reference baseline: naive fixed-window
# ----------------------------------------------------------------------------

def apply_fixed_window_baseline(rep, k):
    """Keeps only the k most recent mentions per (patient, concept_type),
    per the proposal's requirement to isolate provenance-aware compression
    from compression in general. Applied directly to the UNCOMPRESSED
    representation, not to the aggregated one, since this baseline is
    mention-level truncation, not fact-level merging."""
    mention_nodes = [n for n in rep["nodes"] if n["kind"] == "mention"]
    other_nodes = [n for n in rep["nodes"] if n["kind"] != "mention"]
    by_key = defaultdict(list)
    for n in mention_nodes:
        by_key[(n["patient_id"], n["concept_type"])].append(n)

    kept = []
    for _, mentions in by_key.items():
        kept += sorted(mentions, key=lambda m: -m["day"])[:k]
    kept_ids = {n["id"] for n in kept}
    new_edges = [e for e in rep["edges"]
                 if not (e["type"] == "derived_from" and e["from"] not in kept_ids)]
    return {"nodes": other_nodes + kept, "edges": new_edges}


# ----------------------------------------------------------------------------
# 9. Querying a compressed representation (no access to raw facts)
# ----------------------------------------------------------------------------

def query_active_value(rep, patient_id, concept_type, query_day):
    """Answers 'what is the active value for this concept as of
    query_day' using ONLY the given representation -- this is the actual
    test of whether compression preserved revision-sensitive
    answerability, not just whether the oracle (which has raw facts) is
    right. Returns a value string, None (no applicable info at all), or
    "ABSTAIN" (an approximated entry exists but query_day falls in its
    genuinely ambiguous window -- see the BEFORE/AFTER/OVERLAP handling
    below)."""
    candidates = []
    unresolved = False

    for n in rep["nodes"]:
        if n.get("patient_id") != patient_id or n.get("concept_type") != concept_type:
            continue
        if n["kind"] in ("state", "mention"):
            entries = [{"day": n.get("day"), "value": n["value"],
                        "approx_relation": n.get("approx_relation"),
                        "approx_anchor_day": n.get("approx_anchor_day")}]
        elif n["kind"] == "revision_state":
            entries = n["revision_history"]
        else:
            continue

        for h in entries:
            if h.get("day") is not None:
                if h["day"] <= query_day:
                    candidates.append((h["day"], h["value"]))
                continue
            rel, anchor = h.get("approx_relation"), h.get("approx_anchor_day")
            if anchor is None:
                unresolved = True
                continue
            if rel == "AFTER" and query_day <= anchor:
                continue  # true_day > anchor >= query_day -- confidently not yet active
            elif rel == "BEFORE" and query_day >= anchor:
                # true_day < anchor <= query_day -- confidently already active;
                # anchor used as a conservative (later-than-true) sort key
                candidates.append((anchor, h["value"]))
            elif rel == "OVERLAP":
                if anchor <= query_day:
                    candidates.append((anchor, h["value"]))
                # else: true_day == anchor > query_day -- confidently not yet active
            else:
                unresolved = True  # genuinely ambiguous given only the Allen relation

    if candidates:
        candidates.sort(key=lambda c: c[0])
        return candidates[-1][1]
    return "ABSTAIN" if unresolved else None


# ----------------------------------------------------------------------------
# 10. Evaluation harness
# ----------------------------------------------------------------------------

def run_ro2_evaluation(data_dir, confidence_threshold, fixed_window_k,
                        n_query_samples_per_patient, seed):
    rng = random.Random(seed)
    facts_df, edges_df, notes_df = load_ro2_inputs(data_dir)

    facts_by_patient = defaultdict(list)
    for _, f in facts_df.iterrows():
        facts_by_patient[f["patient_id"]].append(f.to_dict())

    uncompressed = make_uncompressed_representation(facts_df, edges_df, notes_df, rng)
    aggregated = apply_aggregation(uncompressed)
    revised = apply_revision_representation(aggregated)
    approximated = apply_temporal_approximation(revised, confidence_threshold, rng)
    fixed_window = apply_fixed_window_baseline(uncompressed, fixed_window_k)

    representations = {
        "uncompressed_baseline": uncompressed,
        "aggregation": aggregated,
        "revision_representation": revised,
        "temporal_approximation": approximated,
        "fixed_window_baseline": fixed_window,
    }

    query_points = []
    for pid, flist in facts_by_patient.items():
        min_day, max_day = min(f["day"] for f in flist), max(f["day"] for f in flist) + 3
        for concept_type in ["diagnosis", "treatment"]:
            days = sorted({rng.randint(min_day, max_day) for _ in range(n_query_samples_per_patient)})
            for d in days:
                query_points.append((pid, concept_type, d))

    results = {}
    for name, rep in representations.items():
        n_correct = n_wrong = n_abstain = n_total = 0
        for pid, concept_type, day in query_points:
            oracle_fact_id = evaluate_revision_sensitive_state(facts_by_patient[pid], day).get(concept_type)
            oracle_value = next((f["value"] for f in facts_by_patient[pid]
                                  if f["fact_id"] == oracle_fact_id), None)
            pred = query_active_value(rep, pid, concept_type, day)
            n_total += 1
            if pred == "ABSTAIN":
                n_abstain += 1
            elif pred == oracle_value:
                n_correct += 1
            else:
                n_wrong += 1

        results[name] = {
            "representation_size": representation_size(rep),
            "revision_sensitive_query_accuracy_on_resolved": n_correct / max(1, n_total - n_abstain),
            "coverage": (n_total - n_abstain) / max(1, n_total),
            "n_correct": n_correct, "n_wrong": n_wrong, "n_abstain": n_abstain, "n_total": n_total,
            "provenance_recall": provenance_recall(rep, uncompressed),
        }
    return results, query_points


# ----------------------------------------------------------------------------
# 11. Main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="output")
    parser.add_argument("--confidence_threshold", type=float, default=0.65,
                         help="Mentions/history-entries with simulated extraction confidence "
                              "below this are subject to temporal_approximation.")
    parser.add_argument("--fixed_window_k", type=int, default=2,
                         help="k for the naive fixed-window reference baseline.")
    parser.add_argument("--n_query_samples_per_patient", type=int, default=6)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out_dir", default=None,
                         help="Defaults to --data_dir if unset.")
    args = parser.parse_args()
    out_dir = args.out_dir or args.data_dir

    print(f"Loading generator output from {args.data_dir}...")
    results, query_points = run_ro2_evaluation(
        args.data_dir, args.confidence_threshold, args.fixed_window_k,
        args.n_query_samples_per_patient, args.seed)
    print(f"  {len(query_points)} (patient, concept, query_day) points sampled for evaluation.\n")

    baseline_total = results["uncompressed_baseline"]["representation_size"]["total"]
    print(f"{'representation':<26} {'nodes':>7} {'edges':>7} {'total':>7} {'vs base':>9} "
          f"{'accuracy':>10} {'coverage':>10} {'prov.recall':>12}")
    for name, r in results.items():
        size = r["representation_size"]
        ratio = size["total"] / baseline_total if baseline_total else float("nan")
        print(f"{name:<26} {size['n_nodes']:>7} {size['n_edges']:>7} {size['total']:>7} "
              f"{ratio:>8.2f}x {r['revision_sensitive_query_accuracy_on_resolved']:>10.3f} "
              f"{r['coverage']:>10.3f} {r['provenance_recall']:>12.3f}")

    for name in ("aggregation", "revision_representation"):
        r = results[name]
        if r["revision_sensitive_query_accuracy_on_resolved"] < 0.999 or r["provenance_recall"] < 0.999:
            print(f"\n  [warn] {name} is claimed to be reversible/exact but scored "
                  f"accuracy={r['revision_sensitive_query_accuracy_on_resolved']:.3f} "
                  f"provenance_recall={r['provenance_recall']:.3f} -- if either is below 1.0 "
                  f"this indicates a real bug in the operator, not expected lossy behavior "
                  f"(unlike temporal_approximation, which is allowed to trade accuracy/coverage "
                  f"for size).")

    out_path = os.path.join(out_dir, "ro2_compression_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults written to {out_path}")
    print("\nReminder: temporal_approximation's confidence values are SIMULATED (style-based "
          "heuristic), not from a real extraction model -- see module docstring before citing "
          "these numbers as a finding about real extraction uncertainty.")


if __name__ == "__main__":
    main()
