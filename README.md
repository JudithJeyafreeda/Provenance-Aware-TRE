# Provenance-Aware Timeline Reconstruction (Provenance-Aware-TRE)

Code and synthetic data for the manuscript **"Auditable Clinical Timeline Reconstruction with Provenance-Aware Evidence Graphs"** (J. J. Andrew, CHU Bordeaux).

The study tests three auditability properties of a clinical timeline-reconstruction pipeline on a **fully synthetic** corpus:

1. **Revision-preserving compression.** Evidence Graph operators shrink the representation while keeping revision-sensitive answers and mention links.
2. **Abstention when evidence is unavailable.** A BioClinicalBERT gate and a zero-shot LLM gate are tested on controls in which the temporal evidence has been removed, both with an explicit omission marker and silently.
3. **Abstentions as explicit graph edges.** A temporally versioned provenance graph separates a declined answer from a different answer.

> **Scope.** All data are synthetic. The results describe implementation behaviour, not clinical performance. Do not use this code for clinical decisions.

---

## Repository contents

### Data (committed)

| File | Contents |
|---|---|
| `synthetic_notes_v1.csv` | 3,353 LLM-paraphrased notes for 1,000 patients, with mention links (`mentioned_fact_ids`). 1,000 notes carry a gold temporal relation (`gold_relation`, `gold_evidence_span`, `event1`, `event2`). |
| `synthetic_facts_v1.csv` | 2,220 clinical facts (diagnoses and treatments) with day and revision pointer. |
| `synthetic_revision_edges_v1.csv` | 220 ground-truth revision edges. |
| `synthetic_adversarial_v1.csv` | 2,000 contradiction-injected variants. |
| `synthetic_generation_diagnosis_v1.json` | Generator diagnostics (counts, style distribution, paraphrase success). |
| `results_clinicalBERT.py` | Results of the original hard-gated BERT pipeline. **This is a JSON file despite the `.py` extension.** |
| `result_llm.json` | Results of the original gated LLM pipeline. |

### Scripts

| Script | Role | Manuscript |
|---|---|---|
| `sythetic_data_generation.py` | Synthetic corpus generator (facts, revisions, notes, LLM paraphrase, adversarial variants) | §2.2, Table 1 |
| `compression_operators.py` | Evidence Graph construction; aggregation, revision-representation and temporal-approximation operators; fixed-window baseline; revision-sensitive query evaluation | §2.7, Table 2, Fig. 3 |
| `provenance_temporal_kg.py` | Temporally versioned provenance graph (never-overwrite edges, ABSTAINED edges, discrepancy flags). Imported by the two pipelines below. | §2.9, Table 8 |
| `provenance_first_Bio_Clinical_BERT.py` | Original hard-gated BioClinicalBERT pipeline, manipulation analysis and certification | Tables 3, 9 |
| `provenance_first_LLM.py` | Original gated LLM pipeline (cited span, coverage check, necessity re-queries), manipulation analysis, certification, provenance graph | Tables 3, 8, 9 |
| `make_evidence_unavailable_adversarial.py` | Marker-carrying evidence-unavailable controls (`remove_evidence`, `mask_temporal_cue`) | §2.3, Fig. 2 |
| `make_silent_removal_controls.py` | Marker-free `silent_removal` / `silent_keep` controls (LLM editor plus LLM judge, strict and implied-only tiers) | §2.4 |
| `regenerate_gold_spans.py` | Post-paraphrase gold-span regeneration (LLM relocator plus judge; verified and relation-only tiers) | §2.5 |
| `train_availability_supervised_bert.py`, `_v2.py`, `_v3.py` | Availability-supervised BERT gate. v2 adds `--save_dir`, `--load_dir` and `--extra_controls`; v3 adds `--export_predicted_spans` and token-overlap recall against regenerated gold spans. Default behaviour is identical across versions. | §2.6, Tables 4–7 |
| `llm_availability_gated_inference_v1.py`, `_v2.py` | LLM availability gate. v2 adds the EVENT1/EVENT2 definitions to the prompt (`--event_mode convention`) and exports the cited span per row. | §2.6, Tables 4–7 |
| `compute_llm_evidence_recall.py` | LLM character-span recall, precision and cited-span length against the regenerated gold spans (no model calls) | Table 4, Fig. 4 |
| `evaluate_control_abstain_v1.py` | Abstention metrics on clean and control predictions | §2.8 |
| `stratify_masked_cue.py` | Paired, stratified abstention results with Wilson 95% intervals: control vs. the same notes when clean, split by control type, masked cue and silent-removal tier (no model calls) | Tables 6–7, Figs. 5–6 |
| `bert_control_inference.py` | Coverage-gated BERT training and control inference (intermediate experiment) | — |
| `bert_prediction_export_v1.py`, `llm_prediction_export_v1.py`, `evaluate_oracle_abstention_metrics.py`, `evaluate_three_claim.py` | Auxiliary and exploratory scripts; not used for the reported results | — |

---

## Requirements

- Python 3.10 or newer
- `pip install numpy pandas scipy networkx torch transformers openai`
- BERT model: `emilyalsentzer/Bio_ClinicalBERT` (downloaded by `transformers`). The encoder is frozen by default; the trainable heads run on CPU or GPU.
- LLM: an **OpenAI-compatible endpoint** (for example vLLM) serving `mistralai/Mistral-Small-3.2-24B-Instruct-2506`. Pass it with `--api_base`; set `VLLM_API_KEY` if your server needs a key.
- `provenance_temporal_kg.py` must be importable, so run the scripts from the repository root.

---

## Reproducing the results

The commands below use the folder layout of the reported run. Set:

```bash
D=output/synthetic_dataset          # all inputs and outputs live here
API=http://localhost:8000/v1         # your OpenAI-compatible endpoint
M=mistralai/Mistral-Small-3.2-24B-Instruct-2506
```

### 0. Corpus

To use the committed corpus (recommended; the reported results use it):

```bash
mkdir -p $D && cp synthetic_*_v1.csv $D/
```

To regenerate it (requires the LLM endpoint; paraphrasing makes it non-identical):

```bash
python sythetic_data_generation.py --n_patients 1000 --use_llm \
  --api_base $API --llm_model $M --seed 7 --out_dir $D
```

The LLM availability gate and the two LLM-edit scripts read the 1,000 relation-eligible notes:

```bash
python -c "import pandas as pd; d=pd.read_csv('$D/synthetic_notes_v1.csv'); d[d.gold_relation.notna()].to_csv('$D/synthetic_notes_relation_only.csv', index=False)"
```

### 1. Evidence Graph compression (Table 2, Fig. 3). Deterministic; no model needed.

```bash
python compression_operators.py --data_dir $D \
  --confidence_threshold 0.65 --fixed_window_k 2 \
  --n_query_samples_per_patient 6 --seed 11
```

Writes `$D/ro2_compression_results.json`. On the committed corpus it reproduces the reported numbers exactly:

| Representation | Nodes | Edges | Size ratio | Query accuracy | Query coverage | Provenance recall |
|---|---|---|---|---|---|---|
| Uncompressed | 6,926 | 7,146 | 1.00 | 1.000 | 1.000 | 1.000 |
| Aggregation | 2,220 | 7,146 | 0.67 | 1.000 | 1.000 | 1.000 |
| Revision-representation | 2,000 | 6,926 | 0.63 | 1.000 | 1.000 | 1.000 |
| Temporal-approximation | 2,000 | 6,978 | 0.64 | 0.987 | 0.996 | 1.000 |
| Fixed-window (k = 2) | 3,982 | 4,202 | 0.58 | 0.453 | 1.000 | 0.575 |

Confidence values in temporal approximation are simulated from note style, not produced by an extraction model.

### 2. Original gated pipelines (Tables 3, 8, 9)

```bash
python provenance_first_Bio_Clinical_BERT.py --data_dir $D --out_dir output \
  --epochs 20 --batch_size 8 --lr 5e-5 --seed 7 --run_certification \
  --cert_n_examples 30 --cert_n_samples 100 --cert_alpha 0.01

python provenance_first_LLM.py --data_dir $D --out_dir output \
  --api_base $API --llm_model $M --temperature 0.0 --seed 7 \
  --n_test 60 --n_manip_pairs 60 --run_certification \
  --cert_n_examples 20 --cert_n_samples 100 --cert_alpha 0.05
```

These write `output/poc_results_clinicalbert_v6.json` and `output/poc_results_llm_v7.json`; the committed `results_clinicalBERT.py` and `result_llm.json` are the reported runs. With 100 perturbations, BERT certification at α = 0.01 is unattainable (the Clopper–Pearson bound for 100/100 is 0.948, below 0.99); 528 perturbations per example would be needed. The evidence-recall value of 0.000 in these files is an artefact: gold spans were recorded before paraphrasing (see §3.3 of the manuscript and step 5 below).

### 3. Evidence-unavailable controls

Marker-carrying (§2.3). Deterministic:

```bash
python make_evidence_unavailable_adversarial.py --notes $D/synthetic_notes_v1.csv \
  --out_csv $D/evidence_removed_only.csv --variants remove_evidence,mask_temporal_cue --seed 7
```

This gives 1,000 `remove_evidence` rows (one distinct text) and 1,000 `mask_temporal_cue` rows (421 with a mask token, 579 kept with an appended omission notice).

Marker-free (§2.4). Uses the LLM; the outputs are not bit-reproducible:

```bash
python make_silent_removal_controls.py --notes_csv $D/synthetic_notes_relation_only.csv \
  --out_dir $D/silent_controls --api_base $API --model $M --workers 8
```

Writes `silent_removal_controls.csv` (507 accepted: 238 strict, 269 implied-only), `silent_keep_controls.csv` (452) and a report. Use `--judge_model` to judge with a different model from the one evaluated as a gate.

### 4. Gold-span regeneration (§2.5). Uses the LLM.

```bash
python regenerate_gold_spans.py --notes_csv $D/synthetic_notes_relation_only.csv \
  --out_dir $D/gold_spans_v2 --api_base $API --model $M --workers 8
```

Reported run: 482 accepted (466 verified, 16 relation-only), 139 judged lost in paraphrase, 379 attempts exhausted (`gold_spans_report.json`).

### 5. Availability-supervised BERT gate (Tables 4–7)

```bash
# train once, save weights, score the clean notes, the marker-carrying controls and the silent controls
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

The patient-level test partition (seed 7) has 150 clean notes, 300 marker-carrying controls (61 masked, 89 with a notice, 150 `remove_evidence`), 81 silent-removal notes and 70 silent-keep notes. The script writes `regular_bert_*` (ungated) and `provenance_first_availability_supervised_*` (gated) prediction CSVs; v3 adds `predicted_evidence_span` and `evidence_recall_v2`. Reported recall on the 77 scorable notes: 0.0485 ungated, 0.0055 gated.

### 6. LLM availability gate (Tables 4–7)

```bash
for f in synthetic_notes_relation_only evidence_removed_only \
         silent_controls/silent_removal_controls silent_controls/silent_keep_controls; do
  python llm_availability_gated_inference_v2.py --input_csv $D/$f.csv \
    --out_dir $D/llm_v2 --api_base $API --model $M --event_mode convention
done
```

The reported abstention run (`llm_v2/`) answered 700 of 1,000 clean notes. The evidence-span recall analysis used a second run of the clean notes (`--out_dir $D/llm_v2_spans`), which answered 694 (accuracy 0.663 in both). The difference reflects non-deterministic inference at temperature 0.0.

The comparison run without event definitions (Table 5) used `llm_availability_gated_inference_v1.py --input_csv $D/synthetic_notes_relation_only.csv --out_dir $D/llm_availability`.

LLM evidence-span recall (no model calls):

```bash
python compute_llm_evidence_recall.py \
  --predictions $D/llm_v2_spans/llm_availability_gated_synthetic_notes_relation_only_predictions.csv \
  --gold_spans_v2 $D/gold_spans_v2/gold_spans_v2.csv \
  --out_csv $D/llm_v2_spans/llm_evidence_recall_v2.csv
```

Abstained rows that cited a span are scored on that span in the pooled figure; add `--exclude_abstained` to score answered rows only. Reported: recall 0.750 pooled (482 rows), and on the 379 answered rows recall 0.850, precision 0.864, cited span 0.339 of the note.

### 7. Paired abstention tables (Tables 6–7, Figs. 5–6). No model calls.

```bash
mkdir -p $D/stratified_all
G=$D/availability_supervised_bert_v2/provenance_first_availability_supervised
# BERT, marker-carrying controls
python stratify_masked_cue.py --control_csv $D/evidence_removed_only.csv \
  --control_predictions ${G}_unavailable_test_predictions.csv \
  --clean_predictions ${G}_clean_test_predictions.csv --out_csv $D/stratified_all/bert_maskremove.csv
# BERT, silent controls (repeat with silent_keep)
python stratify_masked_cue.py --control_csv $D/silent_controls/silent_removal_controls.csv \
  --control_predictions ${G}_silent_removal_controls_test_predictions.csv \
  --clean_predictions ${G}_clean_test_predictions.csv --out_csv $D/stratified_all/bert_silent_removal.csv
# LLM, marker-carrying controls (repeat with the silent_removal / silent_keep prediction files)
python stratify_masked_cue.py --control_csv $D/evidence_removed_only.csv \
  --control_predictions $D/llm_v2/llm_availability_gated_evidence_removed_only_predictions.csv \
  --clean_predictions $D/llm_v2/llm_availability_gated_synthetic_notes_relation_only_predictions.csv \
  --out_csv $D/stratified_all/llm_maskremove.csv
```

Each row pairs a control stratum with the same base notes when clean and reports abstention recall, unsafe-answer rate and answered accuracy, with Wilson 95% intervals.

---

## Reproducibility notes

- **Bit-reproducible from the committed data:** corpus statistics, compression (step 1) and the marker-carrying controls (step 3).
- **Not bit-reproducible:** anything involving the LLM (paraphrasing, silent controls, gold-span regeneration, LLM gates). Re-runs give close but not identical counts even at temperature 0.0.
- **Seeds:** BERT pipelines and splits use seed 7; compression query sampling uses seed 11.
- **Same model in several roles:** the silent-control editor and judge, the gold-span relocator and judge, and the LLM gate were all Mistral-Small-3.2-24B in the reported run. An independent judge model is recommended (`--judge_model`).
- **Majority-class rate:** 0.354 (354 of the 1,000 gold relations are `EVENT1-OVERLAP-EVENT2`).

## Citation

If you use this code, please cite the manuscript (citation to be added on publication) and this repository.

## Licence

To be added by the author.
