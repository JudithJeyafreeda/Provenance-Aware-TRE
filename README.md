# Provenance-Aware Timeline Reconstruction (Provenance-Aware-TRE)

Code and synthetic data for the manuscript **"Auditable Clinical Timeline Reconstruction with Provenance-Aware Evidence Graphs"** (J. J. Andrew, CHU Bordeaux).

The code tests three auditability properties of a clinical timeline-reconstruction pipeline on a **fully synthetic** corpus:

1. **Revision-preserving compression.** Evidence Graph operators shrink the representation while keeping revision-sensitive answers and mention links.
2. **Abstention when evidence is unavailable.** A BioClinicalBERT gate and a zero-shot LLM gate are tested on controls in which the temporal evidence has been removed, both with an explicit omission marker and silently.
3. **Abstentions as explicit graph edges.** A temporally versioned provenance graph separates a declined answer from a different answer.

> **Scope.** All data are synthetic. The code demonstrates implementation behaviour, not clinical performance. Do not use it for clinical decisions.

---

## Repository contents

### Data

| File | Contents |
|---|---|
| `synthetic_notes_v1.csv` | 3,353 LLM-paraphrased notes for 1,000 patients, with mention links (`mentioned_fact_ids`). 1,000 notes carry a gold temporal relation (`gold_relation`, `gold_evidence_span`, `event1`, `event2`). |
| `synthetic_facts_v1.csv` | 2,220 clinical facts (diagnoses and treatments) with day and revision pointer. |
| `synthetic_revision_edges_v1.csv` | 220 ground-truth revision edges. |
| `synthetic_adversarial_v1.csv` | 2,000 contradiction-injected variants. |
| `synthetic_generation_diagnosis_v1.json` | Generator diagnostics (counts, style distribution, paraphrase success). |

Only the input corpus is committed. Model outputs, control files, regenerated gold spans and evaluation results are **not** distributed; every one of them is produced by the scripts below.

### Scripts

| Script | Role |
|---|---|
| `sythetic_data_generation.py` | Synthetic corpus generator: facts, revisions, notes, LLM paraphrase, adversarial variants. |
| `compression_operators.py` | Evidence Graph construction; aggregation, revision-representation and temporal-approximation operators; fixed-window baseline; revision-sensitive query evaluation. |
| `provenance_temporal_kg.py` | Temporally versioned provenance graph (never-overwrite edges, `ABSTAINED` edges, discrepancy flags, summary-only aggregation). Imported by the two pipelines below. |
| `provenance_first_Bio_Clinical_BERT.py` | Hard-gated BioClinicalBERT pipeline with an ungated baseline, manipulation analysis and certification. |
| `provenance_first_LLM.py` | Gated LLM pipeline (cited span, coverage check, necessity re-queries) with an ungated baseline, manipulation analysis, certification and provenance-graph ingestion. |
| `make_evidence_unavailable_adversarial.py` | Marker-carrying evidence-unavailable controls (`remove_evidence`, `mask_temporal_cue`). |
| `make_silent_removal_controls.py` | Marker-free `silent_removal` and `silent_keep` controls (LLM editor plus LLM judge; strict and implied-only tiers). |
| `regenerate_gold_spans.py` | Post-paraphrase gold-span regeneration (LLM relocator plus judge; verified and relation-only tiers). |
| `train_availability_supervised_bert.py`, `_v2.py`, `_v3.py` | Availability-supervised BERT gate. v2 adds `--save_dir`, `--load_dir` and `--extra_controls`; v3 adds `--export_predicted_spans` and token-overlap recall against regenerated gold spans. Default behaviour is identical across versions. |
| `llm_availability_gated_inference_v1.py`, `_v2.py` | LLM availability gate. v2 adds EVENT1/EVENT2 definitions to the prompt (`--event_mode convention`) and exports the cited span for each row. |
| `compute_llm_evidence_recall.py` | LLM character-span recall, precision and cited-span length against regenerated gold spans. No model calls. |
| `evaluate_control_abstain_v1.py` | Abstention metrics on clean and control predictions. |
| `stratify_masked_cue.py` | Paired, stratified abstention results with Wilson 95% intervals: each control against the same notes when clean, split by control type, masked cue and silent-removal tier. No model calls. |
| `bert_control_inference.py` | Coverage-gated BERT training and control inference (intermediate experiment). |
| `bert_prediction_export_v1.py`, `llm_prediction_export_v1.py`, `evaluate_oracle_abstention_metrics.py`, `evaluate_three_claim.py` | Auxiliary and exploratory scripts. |

---

## Requirements

- Python 3.10 or newer
- `pip install numpy pandas scipy networkx torch transformers openai`
- BERT model: `emilyalsentzer/Bio_ClinicalBERT`, downloaded by `transformers`. The encoder is frozen by default; the trainable heads run on CPU or GPU.
- LLM: an **OpenAI-compatible endpoint** (for example vLLM) serving `mistralai/Mistral-Small-3.2-24B-Instruct-2506`. Pass it with `--api_base`, and set `VLLM_API_KEY` if your server needs a key.
- Run all scripts from the repository root, so that `provenance_temporal_kg.py` can be imported.

---

## Running the pipeline

Set:

```bash
D=output/synthetic_dataset          # inputs and outputs live here
API=http://localhost:8000/v1         # your OpenAI-compatible endpoint
M=mistralai/Mistral-Small-3.2-24B-Instruct-2506
```

### 0. Corpus

To use the committed corpus (recommended):

```bash
mkdir -p $D && cp synthetic_*_v1.csv $D/
```

To regenerate it instead (requires the LLM endpoint; paraphrasing makes the notes non-identical to the committed ones):

```bash
python sythetic_data_generation.py --n_patients 1000 --use_llm \
  --api_base $API --llm_model $M --seed 7 --out_dir $D
```

Several scripts read only the 1,000 relation-eligible notes:

```bash
python -c "import pandas as pd; d=pd.read_csv('$D/synthetic_notes_v1.csv'); d[d.gold_relation.notna()].to_csv('$D/synthetic_notes_relation_only.csv', index=False)"
```

### 1. Evidence Graph compression

Deterministic; no model needed.

```bash
python compression_operators.py --data_dir $D \
  --confidence_threshold 0.65 --fixed_window_k 2 \
  --n_query_samples_per_patient 6 --seed 11
```

Writes `$D/ro2_compression_results.json`. 

Confidence values in temporal-approximation are simulated from note style, not produced by an extraction model.

### 2. Hard-gated pipelines

```bash
python provenance_first_Bio_Clinical_BERT.py --data_dir $D --out_dir output \
  --epochs 20 --batch_size 8 --lr 5e-5 --seed 7 --run_certification \
  --cert_n_examples 30 --cert_n_samples 100 --cert_alpha 0.01

python provenance_first_LLM.py --data_dir $D --out_dir output \
  --api_base $API --llm_model $M --temperature 0.0 --seed 7 \
  --n_test 60 --n_manip_pairs 60 --run_certification \
  --cert_n_examples 20 --cert_n_samples 100 --cert_alpha 0.05
```

Things to know before reading these outputs:

- **Certification ceiling.** At α = 0.01 with 100 perturbations, BERT certification is unattainable: the Clopper–Pearson lower bound for 100/100 at α/2 is 0.948, below the required 0.99. At least 528 perturbations per example are needed. At α = 0.05 the threshold is reachable from 72 perturbations.
- **Evidence recall of 0.000.** The evidence-recall value in these files is an artefact, not a model result. Gold spans were recorded before paraphrasing, and none of them is a literal substring of its note. Use the regenerated spans of step 4 instead.
- **BERT manipulation pairs.** The BERT manipulation analysis uses adversarial pairs from all relation-eligible notes, not only the held-out split. Of the 1,887 pairs, 1,319 have base notes from training patients, 279 from development and 289 from test.

### 3. Evidence-unavailable controls

**Marker-carrying.** Deterministic:

```bash
python make_evidence_unavailable_adversarial.py --notes $D/synthetic_notes_v1.csv \
  --out_csv $D/evidence_removed_only.csv --variants remove_evidence,mask_temporal_cue --seed 7
```
Every marker-carrying control contains an explicit statement or token of omission.

**Marker-free.** Uses the LLM; the outputs are not bit-reproducible:

```bash
python make_silent_removal_controls.py --notes_csv $D/synthetic_notes_relation_only.csv \
  --out_dir $D/silent_controls --api_base $API --model $M --workers 8
```

Writes `silent_removal_controls.csv`, `silent_keep_controls.csv`, test-split versions of both, a rejection log, an audit sample and `silent_controls_report.json`. `silent_removal` rows carry a `validity_tier` of `strict` or `implied_only`.

By default the same model edits and judges. Because the strict tier is defined by the judge failing to infer the relation, judging with the model you later evaluate as a gate biases that gate's results on the strict tier. Use `--judge_model` (and `--judge_api_base` if needed) to judge with a different model.

### 4. Gold-span regeneration

Uses the LLM:

```bash
python regenerate_gold_spans.py --notes_csv $D/synthetic_notes_relation_only.csv \
  --out_dir $D/gold_spans_v2 --api_base $API --model $M --workers 8
```

Writes `gold_spans_v2.csv`, `gold_spans_v2_test.csv`, a rejection log, an audit sample and `gold_spans_report.json`. Each note gets a `reason` (accepted, lost in paraphrase, or attempts exhausted) and, if accepted, a `validity_tier` of `verified` or `relation_only`. `--judge_model` is available here too.

### 5. Availability-supervised BERT gate

```bash
# train once, save weights, score clean notes, marker-carrying controls and silent controls
python train_availability_supervised_bert_v2.py --data_dir $D \
  --out_dir $D/availability_supervised_bert_v2 --seed 7 \
  --save_dir $D/availability_supervised_bert_v2/weights \
  --extra_controls $D/silent_controls/silent_removal_controls.csv,$D/silent_controls/silent_keep_controls.csv

# evidence-span recall against the regenerated gold spans, reusing the saved weights
python train_availability_supervised_bert_v3.py --data_dir $D \
  --out_dir $D/availability_supervised_bert_v2 \
  --load_dir $D/availability_supervised_bert_v2/weights \
  --export_predicted_spans --gold_spans_v2 $D/gold_spans_v2/gold_spans_v2.csv
```

The patient-level split (seed 7) is 700 / 150 / 150 relation-eligible notes for train, development and test. The test partition has 150 clean notes and 300 marker-carrying controls: 150 `remove_evidence`, plus 61 masked and 89 with a notice. The silent controls are scored on test patients only.

The script writes `regular_bert_*` (ungated) and `provenance_first_availability_supervised_*` (gated) prediction CSVs. v3 adds `predicted_evidence_span` and `evidence_recall_v2`.

The evidence labels used in training come from the pre-paraphrase gold spans. Those are never substrings of the notes, so the evidence head receives no informative supervision. The development split is not used for model selection.

### 6. LLM availability gate

```bash
for f in synthetic_notes_relation_only evidence_removed_only \
         silent_controls/silent_removal_controls silent_controls/silent_keep_controls; do
  python llm_availability_gated_inference_v2.py --input_csv $D/$f.csv \
    --out_dir $D/llm_v2 --api_base $API --model $M --event_mode convention --workers 8
done
```

Inference at temperature 0.0 through vLLM is not fully deterministic. Two runs on the clean notes can differ by a few answered rows.

For a run without the EVENT1/EVENT2 definitions:

```bash
python llm_availability_gated_inference_v1.py \
  --input_csv $D/synthetic_notes_relation_only.csv --out_dir $D/llm_availability \
  --api_base $API --model $M
```

LLM evidence-span recall (no model calls):

```bash
python compute_llm_evidence_recall.py \
  --predictions $D/llm_v2/llm_availability_gated_synthetic_notes_relation_only_predictions.csv \
  --gold_spans_v2 $D/gold_spans_v2/gold_spans_v2.csv \
  --out_csv $D/llm_v2/llm_evidence_recall_v2.csv
```

By default, abstained rows that cited a span are scored on that span, and rows without a span score 0. Add `--exclude_abstained` to score answered rows only.

### 7. Paired abstention tables

No model calls.

```bash
mkdir -p $D/stratified_all
G=$D/availability_supervised_bert_v2/provenance_first_availability_supervised

# BERT, marker-carrying controls
python stratify_masked_cue.py --control_csv $D/evidence_removed_only.csv \
  --control_predictions ${G}_unavailable_test_predictions.csv \
  --clean_predictions ${G}_clean_test_predictions.csv \
  --out_csv $D/stratified_all/bert_maskremove.csv

# BERT, silent controls (repeat with silent_keep)
python stratify_masked_cue.py --control_csv $D/silent_controls/silent_removal_controls.csv \
  --control_predictions ${G}_silent_removal_controls_test_predictions.csv \
  --clean_predictions ${G}_clean_test_predictions.csv \
  --out_csv $D/stratified_all/bert_silent_removal.csv

# LLM, marker-carrying controls (repeat with the silent_removal / silent_keep prediction files)
python stratify_masked_cue.py --control_csv $D/evidence_removed_only.csv \
  --control_predictions $D/llm_v2/llm_availability_gated_evidence_removed_only_predictions.csv \
  --clean_predictions $D/llm_v2/llm_availability_gated_synthetic_notes_relation_only_predictions.csv \
  --out_csv $D/stratified_all/llm_maskremove.csv
```

Each output row pairs a control stratum with the same base notes when clean. It reports coverage, abstention recall, unsafe-answer rate and answered accuracy, with Wilson 95% intervals.

Definitions used throughout:

- **Abstention recall:** the fraction of controls not answered.
- **Unsafe-answer rate:** the fraction of controls answered with a relation different from the original note's gold relation. A correct answer on a control is not counted as unsafe.
- **Answered accuracy:** correct answers divided by answers. It is undefined when nothing is answered.

---

## Reproducibility notes

- **Bit-reproducible from the committed data:** corpus statistics, compression (step 1), the marker-carrying controls (step 3) and the BERT patient split.
- **Not bit-reproducible:** anything that calls the LLM (paraphrasing, silent controls, gold-span regeneration, LLM gates). Re-runs give close but not identical counts, even at temperature 0.0.
- **Seeds:** BERT pipelines and splits use seed 7; compression query sampling uses seed 11.
- **Same model in several roles:** by default the silent-control editor and judge, the gold-span relocator and judge, and the LLM gate are all the same model. Pass `--judge_model` to use an independent judge.
- **Majority-class rate:** 0.354 (354 of the 1,000 gold relations are `EVENT1-OVERLAP-EVENT2`).

