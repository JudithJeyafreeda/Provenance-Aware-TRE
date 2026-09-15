"""
Provenance Temporal Knowledge Graph (axis-3 implementation)
==============================================================

Implements the third novelty axis described in the roadmap manuscript:
provenance as an object that persists past a single prediction, into a
multi-scale, temporally-versioned, federation-ready knowledge graph.

This module is model-agnostic: it consumes the output shapes already
produced by BOTH proof-of-concept scripts --

  - provenance_first_poc_clinicalBERT.py's model.gated_forward(...) results
  - provenance_first_poc_llm.py's gated_llm_call(...) / baseline_llm_call(...)
    results

-- via two small adapter functions, so the same graph structure and the
same audit/versioning/federation queries work regardless of which model
produced the evidence trace.

Design choices, matched to Section 5 of the manuscript:

  Multi-scale       -> node `scale` attribute (event / episode / trajectory);
                        only `event` is populated by the current POCs, but
                        the schema supports the other two scales unchanged.
  Federated-ready    -> every node/edge carries `site_id`; aggregate_temporal
                        _motifs() computes ONLY summary statistics that would
                        cross a federation boundary (relation-type counts,
                        mean coverage, abstention rate) -- raw text and
                        provenance spans never leave this function. Actual
                        cross-site secure aggregation is NOT implemented;
                        this provides the local per-site structure a
                        federation layer would operate on.
  Temporally         -> edges are NEVER overwritten. Re-extraction of the
  versioned             same relation for the same event pair creates a NEW
                        edge; if it disagrees with an existing active edge,
                        both are cross-flagged (`contradicts` /
                        `contradicted_by`) rather than one replacing the
                        other. Explicit supersede_edge() lets a human
                        reviewer close out an edge's validity window without
                        deleting it. query_as_of() answers "what did we
                        believe at time T?" directly.
  Provenance-first   -> every edge stores the gate's own diagnostics
                        verbatim (coverage, necessity_drop / necessity_
                        satisfied, evidence indices/spans, abstention reason,
                        model_id). Abstained relations are stored as an
                        explicit `ABSTAINED` edge type rather than dropped.
  Entity resolution  -> get_or_create_event_node() keys nodes by
                        (note_id, label, scale) so that repeated/updated
                        extractions of the SAME underlying event resolve to
                        the SAME node. Without this, re-extractions would
                        each create a fresh node pair and contradiction
                        detection would silently never trigger (this was
                        caught and fixed during testing -- always resolve
                        event identity before adding a relation edge).

Requirements:
    pip install networkx numpy

Usage
-----
Run standalone for a self-contained demo:

    python provenance_temporal_kg.py

Or import ingest_bert_result / ingest_llm_result into either POC script to
persist real (not mock) gate outputs as they are produced.
"""

import uuid
from datetime import datetime, timezone

import numpy as np
import networkx as nx


class ProvenanceTemporalKG:
    def __init__(self, site_id="default_site"):
        self.graph = nx.MultiDiGraph()
        self.site_id = site_id
        self._node_index = {}  # (note_id, label, scale) -> node_id

    def _new_id(self, prefix):
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    def get_or_create_event_node(self, label, scale="event", note_id=None, extra=None):
        """Entity resolution: re-extractions of the same event (same
        note_id + label + scale) resolve to the SAME node -- required for
        contradiction detection across repeated/updated extractions to
        function at all. A production system would use a more robust
        resolver (embeddings, normalized clinical entity IDs); this keying
        is sufficient for the POC."""
        key = (note_id, label, scale)
        if key in self._node_index:
            return self._node_index[key]
        node_id = self._new_id(scale)
        self.graph.add_node(
            node_id, label=label, scale=scale, note_id=note_id, site_id=self.site_id,
            created_at=datetime.now(timezone.utc).isoformat(), **(extra or {}),
        )
        self._node_index[key] = node_id
        return node_id

    def add_event_node(self, label, scale="event", note_id=None, extra=None):
        """Alias to get_or_create_event_node -- kept as the primary public
        name used by the ingest adapters below."""
        return self.get_or_create_event_node(label, scale, note_id, extra)

    def _find_active_edges(self, source_id, target_id):
        results = []
        if self.graph.has_edge(source_id, target_id):
            for key, data in self.graph[source_id][target_id].items():
                if data.get("valid_to") is None:
                    results.append((key, data))
        return results

    def add_relation_edge(self, source_id, target_id, relation_type, provenance,
                           valid_from=None, model_id="unknown"):
        """provenance: dict with gate diagnostics and evidence trace, stored
        verbatim. See ingest_bert_result / ingest_llm_result below for how
        each script's native output is mapped into this dict."""
        valid_from = valid_from or datetime.now(timezone.utc).isoformat()
        edge_key = self._new_id("edge")

        active = self._find_active_edges(source_id, target_id)
        contradicts = [k for k, d in active if d.get("relation_type") != relation_type]

        self.graph.add_edge(
            source_id, target_id, key=edge_key,
            relation_type=relation_type, valid_from=valid_from, valid_to=None,
            site_id=self.site_id, model_id=model_id, provenance=provenance,
            contradicts=contradicts,
        )

        for k, d in active:
            if d.get("relation_type") != relation_type:
                self.graph[source_id][target_id][k]["contradicted_by"] = (
                    self.graph[source_id][target_id][k].get("contradicted_by", []) + [edge_key]
                )
        return edge_key

    def supersede_edge(self, source_id, target_id, edge_key, valid_to=None):
        """Temporal versioning: mark an edge as no longer current WITHOUT
        deleting it. History remains fully queryable via audit_trail() /
        query_as_of()."""
        self.graph[source_id][target_id][edge_key]["valid_to"] = (
            valid_to or datetime.now(timezone.utc).isoformat()
        )

    def get_active_relations(self, source_id, target_id):
        return self._find_active_edges(source_id, target_id)

    def audit_trail(self, source_id, target_id):
        """Every historical version of the relation between two nodes, each
        with its full provenance trace."""
        if not self.graph.has_edge(source_id, target_id):
            return []
        return [{"edge_key": k, **d} for k, d in self.graph[source_id][target_id].items()]

    def query_as_of(self, source_id, target_id, as_of_timestamp):
        """Answers 'what was the knowledge state at time T?' (Section 5.3:
        'what were the diagnostic criteria for this condition in 2022?'),
        generalised here to any relation edge."""
        if not self.graph.has_edge(source_id, target_id):
            return []
        return [
            (k, d) for k, d in self.graph[source_id][target_id].items()
            if d["valid_from"] <= as_of_timestamp and (d["valid_to"] is None or d["valid_to"] > as_of_timestamp)
        ]


def aggregate_temporal_motifs(kg):
    """Federated-ready aggregation. Computes ONLY summary statistics that
    could cross a federation boundary without any raw text, evidence span,
    or patient-level detail leaving the site (Section 5.2: 'share temporal
    patterns, not data')."""
    counts, coverages = {}, []
    n_abstained, n_total = 0, 0
    for u, v, k, d in kg.graph.edges(keys=True, data=True):
        if d.get("valid_to") is not None:
            continue
        n_total += 1
        if d["relation_type"] == "ABSTAINED":
            n_abstained += 1
            continue
        counts[d["relation_type"]] = counts.get(d["relation_type"], 0) + 1
        cov = d["provenance"].get("coverage")
        if cov is not None:
            coverages.append(cov)
    return {
        "site_id": kg.site_id, "n_total_edges": n_total, "n_abstained": n_abstained,
        "abstention_rate": n_abstained / max(1, n_total),
        "relation_type_counts": counts,
        "mean_coverage": float(np.mean(coverages)) if coverages else None,
    }


def federate_aggregate_statistics(site_summaries):
    """Combine ALREADY-AGGREGATED per-site statistics (never raw graphs)
    into a population-level summary. In a real deployment this would run
    behind secure aggregation; here it is a plain sum/weighted-mean to make
    the intended data flow explicit and auditable."""
    total_edges = sum(s["n_total_edges"] for s in site_summaries)
    total_abstained = sum(s["n_abstained"] for s in site_summaries)
    combined_counts = {}
    for s in site_summaries:
        for rel, c in s["relation_type_counts"].items():
            combined_counts[rel] = combined_counts.get(rel, 0) + c
    weighted_coverage = [
        (s["mean_coverage"], s["n_total_edges"] - s["n_abstained"])
        for s in site_summaries if s["mean_coverage"] is not None
    ]
    mean_coverage = (
        sum(c * n for c, n in weighted_coverage) / sum(n for _, n in weighted_coverage)
        if weighted_coverage else None
    )
    return {
        "n_sites": len(site_summaries), "total_edges": total_edges,
        "total_abstained": total_abstained,
        "population_abstention_rate": total_abstained / max(1, total_edges),
        "combined_relation_type_counts": combined_counts,
        "population_mean_coverage": mean_coverage,
    }


# ----------------------------------------------------------------------------
# Adapters: consume each POC script's native gate-output shape
# ----------------------------------------------------------------------------

def ingest_bert_result(kg, note_id, event1_label, event2_label, bert_result,
                        tau=0.5, model_id="bio_clinicalbert"):
    """Adapter for provenance_first_poc_clinicalBERT.py's
    model.gated_forward(input_ids, attention_mask)[i] output."""
    e1 = kg.add_event_node(event1_label, scale="event", note_id=note_id)
    e2 = kg.add_event_node(event2_label, scale="event", note_id=note_id)

    if bert_result["abstain"]:
        provenance = {
            "abstained": True, "reason": bert_result.get("reason"),
            "coverage": bert_result.get("coverage"), "source_note_id": note_id,
        }
        edge_key = kg.add_relation_edge(e1, e2, "ABSTAINED", provenance, model_id=model_id)
        return e1, e2, edge_key, None

    probs = bert_result["probs"]
    relation_idx = int(probs.argmax()) if hasattr(probs, "argmax") else int(np.argmax(probs))
    ev_mask = bert_result["ev_mask"]
    ev_mask_list = ev_mask.tolist() if hasattr(ev_mask, "tolist") else list(ev_mask)
    evidence_token_indices = [i for i, v in enumerate(ev_mask_list) if v > 0]

    provenance = {
        "abstained": False, "coverage": float(bert_result.get("coverage", 0.0)),
        "necessity_drop": bert_result.get("necessity_drop"),
        "evidence_token_indices": evidence_token_indices,
        "source_note_id": note_id, "gate_tau": tau,
    }
    edge_key = kg.add_relation_edge(e1, e2, f"REL_IDX_{relation_idx}", provenance, model_id=model_id)
    return e1, e2, edge_key, relation_idx


def ingest_llm_result(kg, note_id, event1_label, event2_label, llm_result,
                       model_id="mistral_small_3.2"):
    """Adapter for provenance_first_poc_llm.py's gated_llm_call(...) /
    baseline_llm_call(...) output."""
    e1 = kg.add_event_node(event1_label, scale="event", note_id=note_id)
    e2 = kg.add_event_node(event2_label, scale="event", note_id=note_id)

    if llm_result["abstain"]:
        provenance = {
            "abstained": True, "reason": llm_result.get("reason"),
            "coverage": llm_result.get("coverage_frac"), "source_note_id": note_id,
            "evidence_spans_attempted": llm_result.get("evidence_spans", []),
        }
        edge_key = kg.add_relation_edge(e1, e2, "ABSTAINED", provenance, model_id=model_id)
        return e1, e2, edge_key, None

    provenance = {
        "abstained": False, "coverage": llm_result.get("coverage_frac"),
        "evidence_spans": llm_result.get("evidence_spans"),
        "evidence_char_ranges": llm_result.get("evidence_ranges"),
        "confidence": llm_result.get("confidence"),
        "necessity_satisfied": llm_result.get("necessity_satisfied"),
        "source_note_id": note_id,
    }
    edge_key = kg.add_relation_edge(e1, e2, llm_result["relation"], provenance, model_id=model_id)
    return e1, e2, edge_key, llm_result["relation"]


# ----------------------------------------------------------------------------
# Self-contained demo
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Demo: axis-3 provenance temporal KG ===\n")

    # --- Site A: BERT-derived extraction ---
    kg_bert = ProvenanceTemporalKG(site_id="site_A_bert")
    mock_bert_result = {
        "abstain": False, "probs": np.array([0.1, 0.85, 0.05]), "coverage": 0.62,
        "ev_mask": np.array([0, 0, 1, 1, 1, 0, 0]), "necessity_drop": 0.31,
    }
    e1, e2, ek1, rel1 = ingest_bert_result(
        kg_bert, "BEFORE_0001_base", "diagnosed with COPD exacerbation",
        "lumbar puncture and antibiotics initiated", mock_bert_result)
    print(f"[site_A_bert] extracted relation index {rel1}, edge={ek1}")

    # --- Site B: LLM-derived extraction of a different note (independent site) ---
    kg_llm = ProvenanceTemporalKG(site_id="site_B_llm")
    mock_llm_result = {
        "abstain": False, "relation": "EVENT1-OVERLAP-EVENT2", "coverage_frac": 0.55,
        "evidence_spans": ["diagnosed with pneumonia on day 3"],
        "evidence_ranges": [(10, 45)], "confidence": 0.78, "necessity_satisfied": True,
    }
    e1b, e2b, ek2, rel2 = ingest_llm_result(
        kg_llm, "OVERLAP_0007_base", "diagnosed with pneumonia",
        "ceftriaxone initiated", mock_llm_result)
    print(f"[site_B_llm] extracted relation {rel2}, edge={ek2}")

    # --- Contradiction demo: re-extract the SAME event pair (same note_id +
    #     labels) at site A with a model update that disagrees with v1 ---
    mock_bert_result_v2 = {
        "abstain": False, "probs": np.array([0.90, 0.05, 0.05]), "coverage": 0.71,
        "ev_mask": np.array([0, 1, 1, 1, 1, 1, 0]), "necessity_drop": 0.40,
    }
    _, _, ek3, rel3 = ingest_bert_result(
        kg_bert, "BEFORE_0001_base", "diagnosed with COPD exacerbation",
        "lumbar puncture and antibiotics initiated", mock_bert_result_v2, model_id="bio_clinicalbert_v2")

    print(f"\n[site_A_bert] re-extraction (v2 model) gives relation index {rel3} (edge={ek3}, "
          f"v1 was {rel1} -- these disagree, as intended for this demo)")
    print("Active relations for this event pair at site_A_bert now (both should appear, cross-flagged):")
    for k, d in kg_bert.get_active_relations(e1, e2):
        print(f"  {k}: {d['relation_type']} contradicts={d.get('contradicts')} "
              f"contradicted_by={d.get('contradicted_by')}")

    print("\nFull audit trail (nothing was deleted or overwritten):")
    for entry in kg_bert.audit_trail(e1, e2):
        print(f"  {entry['edge_key']}: {entry['relation_type']} "
              f"valid_from={entry['valid_from'][:19]} valid_to={entry['valid_to']}")

    # --- Resolve the contradiction explicitly (e.g. after human review) ---
    kg_bert.supersede_edge(e1, e2, ek1)
    print(f"\nAfter human review supersedes edge {ek1} (the v1 extraction):")
    for entry in kg_bert.audit_trail(e1, e2):
        print(f"  {entry['edge_key']}: {entry['relation_type']} valid_to={entry['valid_to']}")

    # --- Federation: aggregate ONLY statistics from each site, never raw text/provenance ---
    summary_A = aggregate_temporal_motifs(kg_bert)
    summary_B = aggregate_temporal_motifs(kg_llm)
    print(f"\n[federation] site_A_bert local summary: {summary_A}")
    print(f"[federation] site_B_llm local summary: {summary_B}")

    population = federate_aggregate_statistics([summary_A, summary_B])
    print(f"\n[federation] population-level aggregate (raw text never left either site): {population}")
