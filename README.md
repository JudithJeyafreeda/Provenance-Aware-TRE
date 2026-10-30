# Provenance-Aware Timeline Reconstruction (Provenance-Aware-TRE)

Code and synthetic data for the manuscript **"Auditable Clinical Timeline Reconstruction with Provenance-Aware Evidence Graphs"** (J. J. Andrew, CHU Bordeaux).

The code tests three auditability properties of a clinical timeline-reconstruction pipeline on a **fully synthetic** corpus:

1. **Revision-preserving compression.** Evidence Graph operators shrink the representation while keeping revision-sensitive answers and mention links.
2. **Abstention when evidence is unavailable.** A BioClinicalBERT gate and a zero-shot LLM gate are tested on controls in which the temporal evidence has been removed, both with an explicit omission marker and silently.
3. **Abstentions as explicit graph edges.** A temporally versioned provenance graph separates a declined answer from a different answer.

> **Scope.** All data are synthetic. The code demonstrates implementation behaviour, not clinical performance. Do not use it for clinical decisions.

---

The same code and data also support a second article, which evaluates the evidence-availability gates in depth:

> **Omission markers rather than missing evidence drive abstention tests in clinical temporal relation extraction** (J. J. Andrew; submitted to *npj Digital Medicine*). It adds confidence-thresholding baselines, per-class analyses, run-to-run stability, a no-learning shortcut baseline and **re-judging of the marker-free controls by a model from a different family (Qwen)**. Steps 8–10 below reproduce its additional results.


## Repository contents

### Data

| File | Contents |
|---|---|
| `synthetic_notes_v1.csv` | 3,353 LLM-paraphrased notes for 1,000 patients, with mention links (`mentioned_fact_ids`). 1,000 notes carry a gold temporal relation (`gold_relation`, `gold_evidence_span`, `event1`, `event2`). |
| `synthetic_facts_v1.csv` | 2,220 clinical facts (diagnoses and treatments) with day and revision pointer. |
| `synthetic_revision_edges_v1.csv` | 220 ground-truth revision edges. |
| `synthetic_adversarial_v1.csv` | 2,000 contradiction-injected variants. |
| `synthetic_generation_diagnosis_v1.json` | Generator diagnostics (counts, style distribution, paraphrase success). |

Only the input corpus is committed. Model outputs, control files, judge outputs, regenerated gold spans and evaluation results are not distributed in this repository; every one of them is produced by the scripts below, and they are available from the author on reasonable request. Steps that call an LLM are not bit-reproducible (see Reproducibility notes), so regenerated outputs give close but not identical counts; `verify_paper_numbers.py` checks the article's numbers against a given set of outputs.

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
| `selective_baselines.py` | Confidence-thresholding baselines: thresholds a score to the gate's clean coverage and reports abstention and AUROC (bootstrap 95% intervals) for each control. No model calls. |
| `per_class_silent.py` | Per-gold-class breakdown of `silent_removal` and `silent_keep` results, paired with the same notes when clean. No model calls. |
| `rejudge_silent_controls.py` | **Independent judge.** Re-judges every accepted `silent_removal` edit with another model (e.g. Qwen), using exactly the judge prompt, inputs and tier rule of `make_silent_removal_controls.py`; prints the tier crosstab, agreement and Cohen's kappa. Used for the cross-family results of the *npj* article. |
| `prepare_judge_input.py` | Builds `notes.jsonl` (clean note, both edits and the original tier per note) for `Judge.py`. |
| `Judge.py` | Supplementary independent judge with its own three-way prompt (explicit / implied / none). Also judges the clean notes (pre-filter) and the `silent_keep` edits, and writes ID lists of the notes it validates. Its prompt differs from the original judge, so tier agreement from this script is **not** comparable with the original judge; use `rejudge_silent_controls.py` for that. |
| `qwen_subset_eval.py` | Scores both gates on the ID lists written by `Judge.py report`, paired with the same notes when clean. No model calls. |
| `shortcut_baseline.py` | No-learning shortcut rule (one day mention → overlap; treatment mentioned first → EVENT2-BEFORE-EVENT1; otherwise EVENT1-BEFORE-EVENT2). No model calls. |
| `check_notice_truncation.py` | Checks whether the appended omission notice survives BERT's 128-token truncation for each notice-appended control. |
| `verify_paper_numbers.py` | Recomputes every data-derived number in the *npj* article from the saved outputs and compares it with the printed value (OK / MISMATCH / MISSING). No model calls. |

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


### 8. Confidence baselines and per-class analysis (*npj* article)

No model calls. The baselines need scores the original exports do not contain, so export them once from the saved BERT weights (no retraining):

```bash
python train_availability_supervised_bert_v3.py --data_dir $D \
  --out_dir $D/availability_supervised_bert_v3_export \
  --load_dir $D/availability_supervised_bert_v2/weights \
  --extra_controls $D/silent_controls/silent_removal_controls.csv,$D/silent_controls/silent_keep_controls.csv
B=$D/availability_supervised_bert_v3_export
```

This writes `argmax_relation` and `max_relation_prob` for both models (`regular_bert_*` = no availability loss). The LLM baselines use a second run of the gate with `--always_probe 1` (the default), which records `llm_confidence` and `reask_agreement` for every non-null answer; in the article this run is `llm_conf`.

```bash
mkdir -p $D/baselines $D/stratified_all
GOLD="$D/synthetic_notes_relation_only.csv $D/silent_controls/silent_removal_controls.csv"
# BERT availability score (repeat with --score_col max_relation_prob, and with the regular_bert_* files)
python selective_baselines.py --clean $B/provenance_first_availability_supervised_clean_test_predictions.csv \
  --condition marker=$B/provenance_first_availability_supervised_unavailable_test_predictions.csv \
  --condition removal=$B/provenance_first_availability_supervised_silent_removal_controls_test_predictions.csv \
  --condition removal_strict=$B/provenance_first_availability_supervised_silent_removal_controls_test_predictions.csv \
  --condition keep=$B/provenance_first_availability_supervised_silent_keep_controls_test_predictions.csv \
  --gold $GOLD --tier_col validity_tier --tier strict --tier_condition removal_strict \
  --pred_col argmax_relation --score_col availability_probability --target_coverage 0.98667 \
  --out $D/baselines/bert_availability.csv

# LLM verbalised confidence (repeat with --score_col reask_agreement)
C=$D/llm_conf
python selective_baselines.py --clean $C/llm_availability_gated_synthetic_notes_relation_only_predictions.csv \
  --condition marker=$C/llm_availability_gated_evidence_removed_only_predictions.csv \
  --condition removal=$C/llm_availability_gated_silent_removal_controls_predictions.csv \
  --condition removal_strict=$C/llm_availability_gated_silent_removal_controls_predictions.csv \
  --condition keep=$C/llm_availability_gated_silent_keep_controls_predictions.csv \
  --gold $GOLD --tier_col validity_tier --tier strict --tier_condition removal_strict \
  --pred_col raw_relation --score_col llm_confidence --target_coverage 0.700 \
  --out $D/baselines/llm_confidence.csv

# Per-class breakdown (BERT shown; same for the LLM files)
python per_class_silent.py \
  --controls $D/silent_controls/silent_removal_controls.csv $D/silent_controls/silent_keep_controls.csv \
  --control_predictions $B/provenance_first_availability_supervised_silent_removal_controls_test_predictions.csv \
                        $B/provenance_first_availability_supervised_silent_keep_controls_test_predictions.csv \
  --clean_predictions $B/provenance_first_availability_supervised_clean_test_predictions.csv \
  --out $D/stratified_all/bert_per_class.csv
```

### 9. Independent (cross-family) judge (*npj* article)

The original editor, judge and LLM gate are all Mistral. Because the strict tier is defined by the judge failing to infer the relation, the article re-judges every accepted removal with a model from a different family. Serve it on an OpenAI-compatible endpoint (the article used `Qwen/Qwen3.8-27B` with vLLM) and run from the repository root:

```bash
python rejudge_silent_controls.py \
  --controls $D/silent_controls/silent_removal_controls.csv \
  --out_csv  $D/silent_controls/silent_removal_controls_qwen.csv \
  --api_base <qwen endpoint>/v1 --model Qwen/Qwen3.8-27B --workers 8
```

- Same judge prompt, inputs (note plus the two event descriptions) and tier rule as the original judge, at temperature 0. For Qwen3 hybrid-thinking models, thinking is switched off (`--no_think 1`, default) and any `<think>` block is removed before parsing.
- The output keeps the original tier in `validity_tier_judge1` and writes the new one to `validity_tier` (`strict`, `implied_only` or `judge_error`).
- Expected output for the article's run: 0 judge errors; 350 of 507 tiers agree (69.0%, Cohen's κ = 0.40); 219 of 238 original strict notes stay strict; 138 of 269 original implied-only notes become strict.

Gate results on the independent judge's tiers, and on the four combinations of both judges (Table 3 of the article), need no model calls:

```bash
# combined label, e.g. "M:strict|Q:strict" = strict under both judges (the judge-independent core)
python -c "
import pandas as pd
p='$D/silent_controls/silent_removal_controls_qwen.csv'
d=pd.read_csv(p); d['validity_tier']='M:'+d.validity_tier_judge1+'|Q:'+d.validity_tier
d.to_csv(p.replace('_qwen.csv','_qwen_combo.csv'), index=False)"

python stratify_masked_cue.py --control_csv $D/silent_controls/silent_removal_controls_qwen_combo.csv \
  --control_predictions $D/llm_v2/llm_availability_gated_silent_removal_controls_predictions.csv \
  --clean_predictions $D/llm_v2/llm_availability_gated_synthetic_notes_relation_only_predictions.csv \
  --out_csv $D/stratified_all/llm_silent_removal_combo.csv

python stratify_masked_cue.py --control_csv $D/silent_controls/silent_removal_controls_qwen.csv \
  --control_predictions $B/provenance_first_availability_supervised_silent_removal_controls_test_predictions.csv \
  --clean_predictions $B/provenance_first_availability_supervised_clean_test_predictions.csv \
  --out_csv $D/stratified_all/bert_silent_removal_qwen.csv
```

**Optional: `Judge.py`.** A second, broader check with its own explicit/implied/none prompt. It also re-runs the clean-note pre-filter and the `silent_keep` acceptance with the independent judge. Because its prompt differs from the original judge, use it for those two checks, not for tier agreement. A determination whose quoted evidence is not a verbatim substring of the note is reported as `unsupported` and excluded, not counted as strict.

```bash
python prepare_judge_input.py \
  --notes $D/synthetic_notes_v1.csv --notes-id note_id --notes-text text --notes-gold gold_relation \
  --removal $D/silent_controls/silent_removal_controls.csv --removal-id base_note_id --removal-text text --removal-tier validity_tier \
  --keep $D/silent_controls/silent_keep_controls.csv --keep-id base_note_id --keep-text text \
  --judge2 $D/silent_controls/silent_removal_controls_judge2.csv --out notes.jsonl
# expected: 1000 notes, 507 silent_removal edits, 452 silent_keep edits

python Judge.py judge  --input notes.jsonl --out qwen_judgments.jsonl \
  --base-url <qwen endpoint>/v1 --model Qwen/Qwen3.8-27B --workers 8
python Judge.py report --input notes.jsonl --judgments qwen_judgments.jsonl --outdir qwen_validation

python qwen_subset_eval.py --notes $D/synthetic_notes_v1.csv --idlists qwen_validation \
  --llm_dir $D/llm_v2 --bert_dir $B --out_csv $D/stratified_all/qwen_subset_results.csv
```

`Judge.py` caches every answer in the `--out` file, so an interrupted run resumes where it stopped.

### 10. Shortcut baseline, truncation check and number verification (*npj* article)

No model calls (the truncation check downloads the BioClinicalBERT tokenizer).

```bash
python shortcut_baseline.py --data_dir $D
# expected: 0.667 overall, 0.667 on the seed-7 test split, 0.996 on structured-admission notes

python check_notice_truncation.py --controls $D/evidence_removed_only.csv \
  --predictions $B/provenance_first_availability_supervised_unavailable_test_predictions.csv \
  --clean_predictions $B/provenance_first_availability_supervised_clean_test_predictions.csv
# expected: the 3 notice-appended controls that BERT answered all exceed 128 tokens

python verify_paper_numbers.py --data_dir $D \
  --bert_dir $B --baseline_bert_dir $B \
  --llm_dir $D/llm_v2 --llm_rerun_dirs $D/llm_conf,$D/llm_v2_spans --baseline_llm_dir $D/llm_conf \
  --recall_csv $D/llm_v2_spans/llm_evidence_recall_v2.csv \
  --rejudge_csv $D/silent_controls/silent_removal_controls_judge2.csv \
  --qwen_csv $D/silent_controls/silent_removal_controls_qwen.csv \
  --out_csv paper_number_check.csv
```

`verify_paper_numbers.py` prints one row per number (section, quantity, printed value, recomputed value, status). The three runs of the LLM gate are `llm_v2` (first run; Tables 1–3), `llm_conf` (second run, with confidence and re-query scores; Table 5) and `llm_v2_spans` (third run, clean notes only; cited-span analysis). Rows for files a run does not contain are reported as MISSING, not failed.

---

## Reproducibility notes

- **Bit-reproducible from the committed data:** corpus statistics, compression (step 1), the marker-carrying controls (step 3) and the BERT patient split.
- **Not bit-reproducible:** anything that calls the LLM (paraphrasing, silent controls, gold-span regeneration, LLM gates). Re-runs give close but not identical counts, even at temperature 0.0.
- **Seeds:** BERT pipelines and splits use seed 7; compression query sampling uses seed 11.
- **Same model in several roles:** by default the silent-control editor and judge, the gold-span relocator and judge, and the LLM gate are all the same model. Re-running the same judge reproduces 500 of 507 tiers, but a judge from another family (step 9) agrees on only 350 of 507 (κ = 0.40), so report results on the notes both judges accept. `--judge_model` in `make_silent_removal_controls.py` and `regenerate_gold_spans.py` sets a different judge at generation time.
- **LLM run-to-run stability:** three runs of the LLM gate at temperature 0 gave clean coverage 0.700, 0.699 and 0.694, and 98.3–98.7% identical clean decisions between any two runs.
- **Majority-class rate:** 0.354 (354 of the 1,000 gold relations are `EVENT1-OVERLAP-EVENT2`).

## Citation

Pipeline, compression and provenance graph:

> J. J. Andrew. *Auditable Clinical Timeline Reconstruction with Provenance-Aware Evidence Graphs.* arXiv:2610.06177 (2026). https://arxiv.org/abs/2610.06177

Evidence-unavailable controls, independent judge and abstention protocol:

> J. J. Andrew. *Omission markers rather than missing evidence drive abstention tests in clinical temporal relation extraction.*

Archived release: [[https://doi.org/10.5281/zenodo.23809568](https://doi.org/10.5281/zenodo.23810126)](https://doi.org/10.5281/zenodo.23810126)
