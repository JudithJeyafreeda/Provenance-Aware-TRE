"""
Diversified Longitudinal Synthetic Clinical Data Generator
==================================================================


Requirements
------------
    pip install pandas numpy
    pip install openai   # only needed if --use_llm is passed

Usage
-----
    # No API needed -- lexical style variation only:
    python synthetic_longitudinal_generator.py --n_patients 200 --out_dir output

    # With LLM paraphrasing for real surface-form diversity:
    python synthetic_data_generation.py --n_patients 1000 --out_dir "path/to/output/dir" --use_llm --api_base "api_base_path" --llm_model "model_name"
"""
import os
import re
import json
import random
import argparse
from collections import Counter

import numpy as np
import pandas as pd

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ----------------------------------------------------------------------------
# 1. Vocabulary
# ----------------------------------------------------------------------------

CONDITIONS = [
    "type 2 diabetes mellitus", "acute heart failure", "community-acquired pneumonia",
    "chronic kidney disease stage 3", "atrial fibrillation", "hypertensive crisis",
    "acute appendicitis", "sepsis of unclear source", "COPD exacerbation", "deep vein thrombosis",
    "acute pancreatitis", "pulmonary embolism", "cellulitis of the lower extremity",
    "gastrointestinal bleed", "acute ischemic stroke", "hyperosmolar hyperglycemic state",
    "acute cholecystitis", "bacterial meningitis", "myocardial infarction", "acute kidney injury",
]
# Revised/confirmed variants used once a culture or workup narrows the diagnosis.
REVISED_CONDITIONS = {
    "sepsis of unclear source": ["sepsis secondary to UTI", "sepsis secondary to pneumonia",
                                  "sepsis secondary to cholangitis"],
    "community-acquired pneumonia": ["hospital-acquired pneumonia", "aspiration pneumonia"],
    "cellulitis of the lower extremity": ["necrotizing soft tissue infection (ruled in on imaging)"],
    "acute kidney injury": ["acute kidney injury, now attributed to contrast nephropathy",
                              "acute kidney injury secondary to obstructive uropathy"],
}
TREATMENTS = [
    "intravenous furosemide", "empiric ceftriaxone", "insulin sliding scale",
    "anticoagulation with apixaban", "supplemental oxygen therapy", "laparoscopic appendectomy",
    "nebulized bronchodilators", "antihypertensive titration", "fluid resuscitation", "beta-blocker therapy",
    "broad-spectrum antibiotics", "thrombolytic therapy", "packed red blood cell transfusion",
    "insulin infusion protocol", "cholecystectomy", "lumbar puncture and antibiotics",
    "percutaneous coronary intervention", "hemodialysis", "vasopressor support", "surgical debridement",
]
NARROWED_TREATMENTS = [
    "narrow-spectrum piperacillin-tazobactam", "targeted vancomycin therapy",
    "de-escalated to oral levofloxacin", "narrowed to ceftriaxone monotherapy",
    "de-escalated to amoxicillin-clavulanate",
]
DOSES = ["1g IV q8h", "500mg PO BID", "2g IV q24h", "40mg PO once daily", "10 units subcutaneously"]
SYMPTOMS = [
    "shortness of breath", "abdominal pain", "palpitations", "lower extremity edema",
    "productive cough", "fever and chills", "dizziness", "chest tightness", "nausea and vomiting", "fatigue",
    "diaphoresis", "confusion", "hematemesis", "syncope", "jaundice", "back pain",
    "decreased urine output", "headache", "tachycardia", "hypotension",
]
DEPARTMENTS = [
    "Cardiology", "Internal Medicine", "Emergency Department", "Nephrology", "Pulmonology",
    "Gastroenterology", "Neurology", "General Surgery", "Infectious Disease", "Critical Care",
]
CULTURE_RESULTS = [
    "blood cultures grew E. coli", "urine culture returned positive for Klebsiella pneumoniae",
    "sputum culture grew Pseudomonas aeruginosa", "blood cultures grew methicillin-resistant Staphylococcus aureus",
    "cultures remained negative but imaging confirmed the source",
]
REVISION_REASONS = [
    "once culture results returned", "after repeat imaging was obtained",
    "following specialist consultation", "after the patient failed to improve on initial therapy",
    "once additional history was obtained from family",
]
CUES_BEFORE = ["prior to", "before", "preceding"]
CUES_AFTER = ["following", "after", "subsequent to"]
CUES_OVERLAP = ["during", "concurrent with", "while undergoing"]
RELATIONS = ["BEFORE", "AFTER", "OVERLAP"]

NOTE_STYLES = ["terse_icu", "verbose_discharge", "nursing_note", "structured_admission"]

STYLE_SYSTEM_PROMPTS = {
    "terse_icu": (
        "Rewrite the following clinical note in a terse ICU progress-note style: "
        "telegraphic sentences, heavy standard medical abbreviation, minimal connective "
        "words. Preserve every clinical fact, every day/date reference, and every "
        "causal or temporal relationship exactly -- do not add, drop, or reorder facts. "
        "Do not invent new clinical details."
    ),
    "verbose_discharge": (
        "Rewrite the following clinical note in a verbose discharge-summary style: "
        "full grammatical sentences, formal register, explicit connective phrases "
        "(e.g. 'subsequently', 'as a result of'). Preserve every clinical fact, every "
        "day/date reference, and every causal or temporal relationship exactly -- do "
        "not add, drop, or reorder facts. Do not invent new clinical details."
    ),
    "nursing_note": (
        "Rewrite the following clinical note in a bedside nursing-note style: "
        "observation-focused, informal shorthand, present-tense fragments where natural. "
        "Preserve every clinical fact, every day/date reference, and every causal or "
        "temporal relationship exactly -- do not add, drop, or reorder facts. Do not "
        "invent new clinical details."
    ),
    "structured_admission": (
        "Rewrite the following clinical note in a structured admission-note style, "
        "as short problem-list-like statements separated by semicolons rather than "
        "flowing prose. Preserve every clinical fact, every day/date reference, and "
        "every causal or temporal relationship exactly -- do not add, drop, or reorder "
        "facts. Do not invent new clinical details."
    ),
}

# Lightweight, no-API fallback so the script is fully runnable offline.
LEXICAL_STYLE_SUBS = {
    "terse_icu": {"the patient": "pt", "diagnosed with": "dx'd w/", "was initiated": "started",
                  "intravenous": "IV", "day": "d", "with": "w/", "and": "&", "following": "s/p"},
    "nursing_note": {"the patient": "pt", "reported": "c/o", "was initiated": "given",
                     "intravenous": "IV", "monitored": "obs'd"},
    "structured_admission": {},  # handled structurally, see render_structured()
    "verbose_discharge": {},     # base style is already closest to this; no lexical change
}


# ----------------------------------------------------------------------------
# 2. Patient timeline construction (ground-truth revision graph)
# ----------------------------------------------------------------------------

def build_patient_timeline(patient_id, rng, notes_per_patient_range=(3, 6), revision_prob=0.6):
    """Builds one patient's fact graph + the notes that mention it.

    Facts: dicts with fact_id, concept_type ('diagnosis'|'treatment'), value,
    day, revises (fact_id or None), dose (treatments only, or None).
    A fact's evidential state at query day d is computed on the fly by
    evaluate_revision_sensitive_state(), not stored, so the oracle is
    always internally consistent with the fact list.
    """
    dept = rng.choice(DEPARTMENTS)
    condition = rng.choice(CONDITIONS)
    treatment = rng.choice(TREATMENTS)
    symptom = rng.choice(SYMPTOMS)
    day0 = rng.randint(1, 3)

    fid = lambda i: f"{patient_id}_F{i:02d}"
    facts = []
    dx0 = {"fact_id": fid(0), "concept_type": "diagnosis", "value": condition,
           "day": day0, "revises": None, "dose": None}
    tx0 = {"fact_id": fid(1), "concept_type": "treatment", "value": treatment,
           "day": day0, "revises": None, "dose": rng.choice(DOSES)}
    facts += [dx0, tx0]

    has_revision = rng.random() < revision_prob and condition in REVISED_CONDITIONS
    revision_day = None
    reason = None
    if has_revision:
        revision_day = day0 + rng.randint(2, 6)
        reason = rng.choice(REVISION_REASONS)
        culture = rng.choice(CULTURE_RESULTS)
        revised_condition = rng.choice(REVISED_CONDITIONS[condition])
        revised_treatment = rng.choice(NARROWED_TREATMENTS)
        dx1 = {"fact_id": fid(2), "concept_type": "diagnosis", "value": revised_condition,
               "day": revision_day, "revises": dx0["fact_id"], "dose": None, "culture": culture}
        tx1 = {"fact_id": fid(3), "concept_type": "treatment", "value": revised_treatment,
               "day": revision_day, "revises": tx0["fact_id"], "dose": rng.choice(DOSES)}
        facts += [dx1, tx1]

    n_notes = rng.randint(*notes_per_patient_range)
    last_day = (revision_day + rng.randint(1, 4)) if has_revision else (day0 + rng.randint(3, 8))
    # day0 and (if present) revision_day are REQUIRED days -- they must never be dropped,
    # since a dropped revision_day means the revision itself is never rendered into any
    # note's text, silently breaking the ground-truth revision edge (it would still be
    # written to synthetic_revision_edges_v1.csv with no supporting mention anywhere).
    required_days = {day0} | ({revision_day} if has_revision else set())
    n_extra = max(0, n_notes - len(required_days))
    extra_days = {rng.randint(day0, last_day) for _ in range(n_extra)}
    note_days = sorted(required_days | extra_days)
    if len(note_days) > n_notes:
        # trim only non-required days, earliest-first, never touching required_days
        trimmable = sorted(set(note_days) - required_days)
        keep_extra = trimmable[:max(0, n_notes - len(required_days))]
        note_days = sorted(required_days | set(keep_extra))

    notes = []
    for i, day in enumerate(note_days):
        is_admission = (i == 0)
        is_revision_note = has_revision and (day == revision_day)
        notes.append({
            "note_idx": i, "day": day, "is_admission": is_admission,
            "is_revision_note": is_revision_note, "dept": dept, "symptom": symptom,
        })

    return {
        "patient_id": patient_id, "facts": facts, "notes": notes,
        "condition": condition, "treatment": treatment, "symptom": symptom,
        "dept": dept, "has_revision": has_revision, "revision_day": revision_day,
        "reason": reason,
    }


def evaluate_revision_sensitive_state(facts, query_day):
    """Oracle for RO2's revision-sensitive query accuracy metric: returns,
    per concept_type, the fact_id that is the correct 'active' evidential
    state as of query_day, given only the fact list's day/revises fields.
    A fact is active at query_day if fact.day <= query_day AND no other
    fact that revises it has fact.day <= query_day."""
    by_concept = {}
    for f in facts:
        if f["day"] > query_day:
            continue
        superseded = any(
            g["revises"] == f["fact_id"] and g["day"] <= query_day for g in facts
        )
        if not superseded:
            by_concept[f["concept_type"]] = f["fact_id"]
    return by_concept


# ----------------------------------------------------------------------------
# 3. Note rendering (base template text, one sentence per note)
# ----------------------------------------------------------------------------

def render_admission_sentence(timeline, rng):
    """The admission note establishes the initial dx/tx with an explicit
    BEFORE/AFTER/OVERLAP relation, kept compatible with the existing
    ClinicalRelationDataset schema (event1/event2/gold_relation/
    gold_evidence_span) used in v6/v7."""
    condition, treatment, symptom = timeline["condition"], timeline["treatment"], timeline["symptom"]
    day0 = timeline["facts"][0]["day"]
    day2 = day0 + rng.randint(0, 1)
    event1 = f"the patient was diagnosed with {condition}"
    event2 = f"{treatment} was initiated"
    relation_type = rng.choice(RELATIONS)

    if relation_type == "BEFORE":
        cue = rng.choice(CUES_BEFORE)
        sentence = (f"On day {day0}, {event1}. {cue.capitalize()} initiating {treatment} "
                    f"on day {day2}, the patient reported {symptom}.")
        gold_relation = "EVENT1-BEFORE-EVENT2"
    elif relation_type == "AFTER":
        cue = rng.choice(CUES_AFTER)
        sentence = (f"{event2.capitalize()} on day {day0}. {cue.capitalize()} treatment, "
                    f"{event1} on day {day2}, correlating with persistent {symptom}.")
        gold_relation = "EVENT2-BEFORE-EVENT1"
    else:
        cue = rng.choice(CUES_OVERLAP)
        sentence = (f"{event1.capitalize()} on day {day0}, and {cue} the diagnostic workup, "
                    f"{event2} to manage associated {symptom}.")
        gold_relation = "EVENT1-OVERLAP-EVENT2"

    return sentence, gold_relation, event1, event2


def render_revision_sentence(timeline):
    dx0, tx0, dx1, tx1 = timeline["facts"][0], timeline["facts"][1], timeline["facts"][2], timeline["facts"][3]
    return (f"{timeline['reason'].capitalize()} ({dx1['culture']}), the working diagnosis of "
            f"{dx0['value']} was revised to {dx1['value']} on day {dx1['day']}; "
            f"{tx0['value']} was changed to {tx1['value']} accordingly.")


def render_followup_sentence(timeline, note):
    facts_asof = evaluate_revision_sensitive_state(timeline["facts"], note["day"])
    fact_by_id = {f["fact_id"]: f for f in timeline["facts"]}
    dx = fact_by_id.get(facts_asof.get("diagnosis"))
    tx = fact_by_id.get(facts_asof.get("treatment"))
    if dx is None or tx is None:
        return None
    return (f"The patient continued to be managed for {dx['value']} with {tx['value']}, "
            f"and remained stable on the {timeline['dept'].lower()} service on day {note['day']}.")


def render_base_text(timeline, note, rng):
    preamble = f"Admitted to {timeline['dept']} for evaluation. " if note["is_admission"] else ""
    closing = (f" The patient was monitored on the {timeline['dept'].lower()} service and remained "
               f"hemodynamically stable.")
    gold_relation, event1, event2, gold_evidence_span = None, None, None, None
    mentioned_fact_ids = []

    if note["is_admission"]:
        sentence, gold_relation, event1, event2 = render_admission_sentence(timeline, rng)
        gold_evidence_span = sentence
        mentioned_fact_ids = [timeline["facts"][0]["fact_id"], timeline["facts"][1]["fact_id"]]
    elif note["is_revision_note"]:
        sentence = render_revision_sentence(timeline)
        # revision note explicitly restates the old value and states the new one for
        # both diagnosis and treatment, so all four facts are "mentioned" here.
        mentioned_fact_ids = [f["fact_id"] for f in timeline["facts"][:4]]
    else:
        sentence = render_followup_sentence(timeline, note)
        if sentence is None:
            sentence, gold_relation, event1, event2 = render_admission_sentence(timeline, rng)
            gold_evidence_span = sentence
            mentioned_fact_ids = [timeline["facts"][0]["fact_id"], timeline["facts"][1]["fact_id"]]
        else:
            # follow-up notes restate whichever fact is currently active as of this
            # note's day -- reuse the same oracle the note text was generated from,
            # so mention-linkage is guaranteed consistent with what the sentence says.
            facts_asof = evaluate_revision_sensitive_state(timeline["facts"], note["day"])
            mentioned_fact_ids = list(facts_asof.values())

    text = preamble + sentence + closing
    return text, gold_relation, event1, event2, gold_evidence_span, mentioned_fact_ids


# ----------------------------------------------------------------------------
# 4. Style diversification
# ----------------------------------------------------------------------------

def apply_lexical_style(text, style):
    """No-API fallback: deterministic lexical rewrite. Weaker diversity than
    LLM paraphrase but exercises the same downstream code path and is
    always available offline."""
    if style == "structured_admission":
        parts = re.split(r"(?<=[.;])\s+", text.strip())
        return "; ".join(p.rstrip(".;") for p in parts if p.strip()) + "."
    subs = LEXICAL_STYLE_SUBS.get(style, {})
    out = text
    for k, v in subs.items():
        out = re.sub(re.escape(k), v, out, flags=re.IGNORECASE)
    return out


def llm_paraphrase(client, model, text, style, temperature=0.7, max_retries=2, timeout=60):
    """Returns (paraphrased_text, ok). Falls back to the lexical rewrite
    (ok=False) on any failure so the pipeline never blocks on API flakiness
    -- callers should track the ok flag and warn if failures are frequent,
    the same defensive pattern used in call_llm() in the v7 script."""
    system_prompt = STYLE_SYSTEM_PROMPTS[style]
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=temperature, max_tokens=300, timeout=timeout,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
            )
            out = resp.choices[0].message.content
            if out and out.strip():
                return out.strip(), True
        except Exception as e:
            print(f"    [warn] llm_paraphrase attempt {attempt+1} failed: {str(e)[:200]}")
    return apply_lexical_style(text, style), False


# ----------------------------------------------------------------------------
# 5. Contradiction injection (generalized from the single fabricated-clause
#    trick in v6/v7 into several distinct, trackable types)
# ----------------------------------------------------------------------------

CONTRADICTION_TYPES = ["fabricated_symptom", "temporal_contradiction", "dosage_contradiction", "negation_flip"]


def _insert_clause(text, clause):
    cut = text.rfind(".", 0, len(text) - 1)
    if cut == -1:
        cut = len(text) - 1
    new_text = text[:cut] + "." + clause + text[cut + 1:]
    start = cut + 1
    end = start + len(clause)
    return new_text, (start, end)


def inject_contradiction(text, timeline, note, contradiction_type, rng):
    if contradiction_type == "fabricated_symptom":
        other_symptom = rng.choice([s for s in SYMPTOMS if s != timeline["symptom"]])
        clause = f" Notably, the patient also mentioned {other_symptom} which had resolved spontaneously and was not treated."
    elif contradiction_type == "temporal_contradiction":
        wrong_day = max(1, note["day"] + rng.choice([-3, -2, 2, 3]))
        clause = f" A separate note on file instead documents this event as occurring on day {wrong_day}."
    elif contradiction_type == "dosage_contradiction":
        facts_asof = evaluate_revision_sensitive_state(timeline["facts"], note["day"])
        fact_by_id = {f["fact_id"]: f for f in timeline["facts"]}
        tx = fact_by_id.get(facts_asof.get("treatment"))
        other_dose = rng.choice([d for d in DOSES if d != (tx or {}).get("dose")])
        clause = f" The medication administration record instead lists a dose of {other_dose}."
    elif contradiction_type == "negation_flip":
        facts_asof = evaluate_revision_sensitive_state(timeline["facts"], note["day"])
        fact_by_id = {f["fact_id"]: f for f in timeline["facts"]}
        dx = fact_by_id.get(facts_asof.get("diagnosis"))
        clause = f" On further review, no definitive evidence of {(dx or {}).get('value', 'this diagnosis')} was found."
    else:
        raise ValueError(f"unknown contradiction_type: {contradiction_type}")

    new_text, span = _insert_clause(text, clause)
    return new_text, {"type": contradiction_type, "start": span[0], "end": span[1], "clause": clause.strip()}


def make_adversarial_variant(base_row, timeline, note, contradiction_types, density_range, rng):
    text = base_row["text"]
    injected = []
    k = rng.randint(*density_range)
    chosen_types = [rng.choice(contradiction_types) for _ in range(k)]
    for ctype in chosen_types:
        text, span_info = inject_contradiction(text, timeline, note, ctype, rng)
        injected.append(span_info)
    row = dict(base_row)
    row["text"] = text
    row["variant"] = "adversarial"
    row["note_id"] = base_row["note_id"] + "_adv"
    row["base_note_id"] = base_row["note_id"]
    row["injected_spans"] = json.dumps(injected)
    row["contradiction_types"] = json.dumps(chosen_types)
    return row


# ----------------------------------------------------------------------------
# 6. Dataset assembly
# ----------------------------------------------------------------------------

def generate_dataset(n_patients, notes_per_patient_range, revision_prob, styles,
                      use_llm, client, llm_model, contradiction_types, n_adversarial_per_patient,
                      contradiction_density_range, seed=7):
    rng = random.Random(seed)
    note_rows, fact_rows, edge_rows, adv_rows = [], [], [], []
    llm_ok_count, llm_fail_count = 0, 0
    style_counter = Counter()

    for pidx in range(n_patients):
        patient_id = f"P{pidx:05d}"
        timeline = build_patient_timeline(patient_id, rng, notes_per_patient_range, revision_prob)

        for f in timeline["facts"]:
            fact_rows.append({"patient_id": patient_id, **f})
            if f["revises"] is not None:
                edge_rows.append({
                    "patient_id": patient_id, "from_fact_id": f["revises"], "to_fact_id": f["fact_id"],
                    "relation": "revises", "day": f["day"],
                })

        rendered_notes = []
        for note in timeline["notes"]:
            base_text, gold_relation, event1, event2, gold_evidence_span, mentioned_fact_ids = \
                render_base_text(timeline, note, rng)
            style = rng.choice(styles)
            style_counter[style] += 1

            if use_llm and client is not None:
                styled_text, ok = llm_paraphrase(client, llm_model, base_text, style)
                llm_ok_count += int(ok)
                llm_fail_count += int(not ok)
                if not ok:
                    gold_evidence_span = None  # exact-substring gold span no longer valid after fallback rewrite
            else:
                styled_text = apply_lexical_style(base_text, style)

            note_id = f"{patient_id}_N{note['note_idx']:02d}"
            row = {
                "note_id": note_id, "patient_id": patient_id, "day": note["day"],
                "style": style, "is_admission": note["is_admission"],
                "is_revision_note": note["is_revision_note"], "text": styled_text,
                "variant": "base", "gold_relation": gold_relation,
                "gold_evidence_span": gold_evidence_span, "event1": event1, "event2": event2,
                "base_note_id": note_id, "injected_spans": "[]", "contradiction_types": "[]",
                "mentioned_fact_ids": json.dumps(mentioned_fact_ids),
            }
            note_rows.append(row)
            rendered_notes.append((row, note))

        candidates = [(r, n) for r, n in rendered_notes if r["is_admission"] or r["is_revision_note"]]
        if not candidates:
            candidates = rendered_notes
        for _ in range(n_adversarial_per_patient):
            base_row, note = rng.choice(candidates)
            adv_row = make_adversarial_variant(base_row, timeline, note, contradiction_types,
                                                contradiction_density_range, rng)
            adv_rows.append(adv_row)

    df_notes = pd.DataFrame(note_rows)
    df_facts = pd.DataFrame(fact_rows)
    df_edges = pd.DataFrame(edge_rows)
    df_adv = pd.DataFrame(adv_rows)

    diagnostics = {
        "n_patients": n_patients, "n_notes": len(df_notes), "n_facts": len(df_facts),
        "n_revision_edges": len(df_edges), "n_adversarial": len(df_adv),
        "style_distribution": dict(style_counter),
        "llm_paraphrase_ok": llm_ok_count, "llm_paraphrase_fallback": llm_fail_count,
        "frac_patients_with_revision": float(df_facts.groupby("patient_id")["revises"]
                                              .apply(lambda s: s.notna().any()).mean()) if len(df_facts) else 0.0,
    }
    return df_notes, df_facts, df_edges, df_adv, diagnostics


# ----------------------------------------------------------------------------
# 7. Main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_patients", type=int, default=200)
    parser.add_argument("--notes_per_patient_min", type=int, default=3)
    parser.add_argument("--notes_per_patient_max", type=int, default=6)
    parser.add_argument("--revision_prob", type=float, default=0.6,
                         help="Probability a patient's diagnosis is revised mid-timeline "
                              "(only applies to conditions with an entry in REVISED_CONDITIONS).")
    parser.add_argument("--styles", default=",".join(NOTE_STYLES),
                         help="Comma-separated subset of: " + ",".join(NOTE_STYLES))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out_dir", default="output")

    parser.add_argument("--use_llm", action="store_true",
                         help="If set, paraphrase each note via an LLM API call for real "
                              "surface-form diversity. Falls back to lexical rewriting per-call "
                              "on any API failure. If not set, uses lexical rewriting only "
                              "(fully offline).")
    parser.add_argument("--api_base", default="http://localhost:8000/v1")
    parser.add_argument("--api_key", default=os.getenv("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--llm_model", default="mistralai/Mistral-Small-3.2-24B-Instruct-2506")

    parser.add_argument("--contradiction_types", default=",".join(CONTRADICTION_TYPES),
                         help="Comma-separated subset of: " + ",".join(CONTRADICTION_TYPES))
    parser.add_argument("--n_adversarial_per_patient", type=int, default=2)
    parser.add_argument("--contradiction_density_min", type=int, default=1)
    parser.add_argument("--contradiction_density_max", type=int, default=2)

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    contradiction_types = [c.strip() for c in args.contradiction_types.split(",") if c.strip()]
    for s in styles:
        assert s in NOTE_STYLES, f"unknown style {s!r}, expected one of {NOTE_STYLES}"
    for c in contradiction_types:
        assert c in CONTRADICTION_TYPES, f"unknown contradiction type {c!r}, expected one of {CONTRADICTION_TYPES}"

    client = None
    if args.use_llm:
        if OpenAI is None:
            raise RuntimeError("--use_llm was passed but the `openai` package is not installed "
                                "(pip install openai).")
        client = OpenAI(base_url=args.api_base, api_key=args.api_key)
        print(f"[llm] paraphrasing via {args.api_base} model={args.llm_model}")
    else:
        print("[llm] --use_llm not set: using offline lexical style rewriting only.")

    print(f"[1/2] Generating {args.n_patients} patient timelines "
          f"(revision_prob={args.revision_prob}, styles={styles})...")
    df_notes, df_facts, df_edges, df_adv, diag = generate_dataset(
        n_patients=args.n_patients,
        notes_per_patient_range=(args.notes_per_patient_min, args.notes_per_patient_max),
        revision_prob=args.revision_prob, styles=styles,
        use_llm=args.use_llm, client=client, llm_model=args.llm_model,
        contradiction_types=contradiction_types,
        n_adversarial_per_patient=args.n_adversarial_per_patient,
        contradiction_density_range=(args.contradiction_density_min, args.contradiction_density_max),
        seed=args.seed,
    )

    print(f"  notes={diag['n_notes']} facts={diag['n_facts']} revision_edges={diag['n_revision_edges']} "
          f"adversarial={diag['n_adversarial']}")
    print(f"  fraction of patients with a real revision: {diag['frac_patients_with_revision']:.3f}")
    print(f"  style distribution: {diag['style_distribution']}")
    if args.use_llm:
        total = diag["llm_paraphrase_ok"] + diag["llm_paraphrase_fallback"]
        fail_rate = diag["llm_paraphrase_fallback"] / max(1, total)
        print(f"  llm paraphrase: {diag['llm_paraphrase_ok']} ok, {diag['llm_paraphrase_fallback']} "
              f"fell back to lexical rewrite (fail_rate={fail_rate:.3f})")
        if fail_rate > 0.1:
            print("  [warn] >10% of LLM paraphrase calls fell back -- check --api_base/--llm_model "
                  "before trusting style diversity claims from this run.")

    print("[2/2] Writing outputs...")
    df_notes.to_csv(os.path.join(args.out_dir, "synthetic_notes_v1.csv"), index=False)
    df_facts.to_csv(os.path.join(args.out_dir, "synthetic_facts_v1.csv"), index=False)
    df_edges.to_csv(os.path.join(args.out_dir, "synthetic_revision_edges_v1.csv"), index=False)
    df_adv.to_csv(os.path.join(args.out_dir, "synthetic_adversarial_v1.csv"), index=False)
    with open(os.path.join(args.out_dir, "synthetic_generation_diagnostics_v1.json"), "w") as f:
        json.dump(diag, f, indent=2)

    n_relation_rows = df_notes["gold_relation"].notna().sum()
    print(f"\nDone. {n_relation_rows}/{len(df_notes)} notes have a gold_relation "
          f"(admission notes usable directly with the v6/v7 ClinicalRelationDataset schema).")
    print("Files written: synthetic_notes_v1.csv, synthetic_facts_v1.csv, "
          "synthetic_revision_edges_v1.csv, synthetic_adversarial_v1.csv, "
          "synthetic_generation_diagnostics_v1.json")


if __name__ == "__main__":
    main()
