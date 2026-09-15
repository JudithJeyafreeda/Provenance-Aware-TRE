# Provenance-First Clinical NLP Proof of Concept

A reproducible proof-of-concept implementation of a provenance-aware pipeline for longitudinal clinical evidence reconstruction.

The repository contains code for:

1. Synthetic longitudinal clinical-corpus generation.
2. Evidence-faithful temporal relation extraction with BioClinicalBERT.
3. Zero-shot LLM-based temporal relation extraction with provenance gating.
4. Evidence Graph compression and revision-sensitive query evaluation.
5. Robustness and certification experiments.
6. Temporally versioned, contradiction-aware provenance graphs.
7. Cross-model result comparison.

The current evaluation uses a fully synthetic corpus. It is intended to validate pipeline mechanisms and implementation logic, not to establish clinical performance on real patient data.

## Pipeline overview

The pipeline has two largely independent branches.

### Ground-truth compression branch

The synthetic generator produces facts, notes, mention links, and revision edges. These are assembled directly into an Evidence Graph and used to evaluate:

- Aggregation.
- Revision-representation.
- Temporal-approximation.
- A naive fixed-window baseline.

This branch does not use predictions from either extractor. Its purpose is to test whether compression preserves known-correct, revision-sensitive answers.

### Model-prediction branch

The same synthetic corpus is passed through two relation-extraction implementations:

- A fine-tuned or frozen BioClinicalBERT model with a hard evidence gate.
- A zero-shot LLM pipeline with cited evidence spans, coverage checks, and a multi-probe necessity gate.

Their outputs are used for:

- Relation-extraction evaluation.
- Adversarial manipulation testing.
- Robustness certification.
- Provenance-graph construction.
- Cross-model disagreement analysis.

## Repository structure

The principal scripts are:

| Script | Role |
|---|---|
| `synthetic_longitudinal_generator.py` | Generates the synthetic longitudinal corpus, revisions, paraphrased notes, and adversarial variants. |
| `provenance_first_poc_clinicalBERT.py` | BioClinicalBERT extraction, evidence masking, hard gating, robustness evaluation, and certification. |
| `provenance_first_poc_llm.py` | LLM extraction, fuzzy evidence-span localization, multi-probe necessity gating, robustness evaluation, and certification. |
| `ro2_compression_operators.py` | Evidence Graph construction, compression operators, query evaluation, and provenance-recall measurement. |
| `provenance_temporal_kg.py` | Temporally versioned provenance graph, entity resolution, contradiction flagging, and aggregate statistics. |


Generated data and result files are normally stored under an output directory such as:

```text
output/
├── synthetic_dataset/
├── corpus_stats.json
├── poc_results_clinicalbert_v6.json
├── poc_results_llm_v7.json
├── ro2_compression_results.json
├── provenance_kg_bert.pkl
├── provenance_kg_provenancefirst.pkl
└── provenance_kg_llm.pkl
```

The exact filenames may vary with the command used and the version of the scripts.

## Requirements

The repository requires Python 3.10 or newer.

Typical dependencies include:

```bash
pip install numpy pandas scipy scikit-learn networkx
pip install torch transformers
pip install openai
```

Additional packages may be required by the installed model, document-processing workflow, or local inference server.

The BERT pipeline requires access to the BioClinicalBERT model:

```text
emilyalsentzer/Bio_ClinicalBERT
```

The LLM pipeline expects an OpenAI-compatible API endpoint. In the reported run, the endpoint served:

```text
mistralai/Mistral-Small-3.2-24B-Instruct-2506
```

The local environment must also contain:

```text
provenance_temporal_kg.py
```

or make it importable through `PYTHONPATH`.

## Reproducible workflow

Run the components in the following order.

### 1. Generate the corpus

Generate the synthetic corpus once:

```bash
python synthetic_longitudinal_generator_v1.py \
  --n_patients 1000 \
  --use_llm \
  --out_dir output/synthetic_dataset
```

The generator should produce files including:

```text
synthetic_notes_v1.csv
synthetic_facts_v1.csv
synthetic_revision_edges_v1.csv
synthetic_adversarial_v1.csv
```

Run the generator once and reuse the resulting directory. Regenerating the corpus before each model run changes the corpus and can invalidate the identical-split comparison.

The reported corpus contains:

- 1,000 patients.
- 3,353 notes.
- 2,220 clinical facts.
- 220 ground-truth revision edges.
- 2,000 adversarial variants.
- 3,353 successful LLM paraphrases.
- No paraphrase fallbacks in the reported run.
- 11.0% of patients with a documented revision.

### 2. Run the BERT pipeline

A representative command is:

```bash
python provenance_first_poc_clinicalBERT_v6.py \
  --data_dir output/synthetic_dataset \
  --model_name emilyalsentzer/Bio_ClinicalBERT \
  --epochs 20 \
  --batch_size 8 \
  --lr 5e-5 \
  --min_coverage 0.08 \
  --min_coverage_target 0.10 \
  --lambda_evi 0.0005 \
  --lambda_coverage_floor 1.0 \
  --lambda_faith 0.5 \
  --lambda_causal 0.1 \
  --warmup_epochs 6 \
  --cert_n_examples 30 \
  --cert_n_samples 100 \
  --cert_alpha 0.01 \
  --cert_epsilon_budget 2 \
  --cert_theta_stability 0.5 \
  --seed 7 \
  --run_certification
```

The reported BERT configuration used:

```text
finetune_bert=False
min_coverage=0.08
min_coverage_target=0.10
min_coverage_stop_threshold=0.08
lambda_evi=0.0005
lambda_coverage_floor=1.0
warmup_epochs=6
cert_n_samples=100
cert_alpha=0.01
```

Check the final training coverage before interpreting downstream results. The gate is not useful if training drives evidence coverage toward zero.

### 3. Run the LLM pipeline

Start an OpenAI-compatible local or remote inference endpoint, then run:

```bash
python provenance_first_poc_llm_v7.py \
  --data_dir output/synthetic_dataset \
  --api_base http://localhost:8000/v1 \
  --llm_model mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
  --temperature 0.0 \
  --seed 7 \
  --n_test 60 \
  --n_manip_pairs 60 \
  --min_coverage_chars 10 \
  --min_coverage_frac 0.03 \
  --necessity_n_probes 3 \
  --necessity_agreement_threshold 0.5 \
  --fuzzy_span_threshold 0.85 \
  --run_certification \
  --cert_n_examples 20 \
  --cert_n_samples 100 \
  --cert_alpha 0.05 \
  --cert_epsilon_budget 2 \
  --cert_theta_stability 0.5
```

The LLM pipeline uses:

- Exact evidence-span matching first.
- Fuzzy matching when exact matching fails.
- Minimum evidence-character and evidence-fraction thresholds.
- Multiple necessity probes.
- A majority-style necessity agreement threshold.
- Character-range overlap for evidence stability.

The LLM run must use the same corpus directory and seed as the BERT run if the outputs are to be compared directly.

### 4. Run Evidence Graph compression

Run:

```bash
python ro2_compression_operators.py \
  --data_dir output/synthetic_dataset \
  --confidence_threshold 0.65 \
  --fixed_window_k 2 \
  --n_query_samples_per_patient 6 \
  --seed 11
```

This evaluates:

- `uncompressed_baseline`
- `aggregation`
- `revision_representation`
- `temporal_approximation`
- `fixed_window_baseline`

Each representation is scored on:

- Node count.
- Edge count.
- Total representation size.
- Revision-sensitive query accuracy.
- Query coverage.
- Provenance recall.

The reported run generated 6,813 evaluation points across 1,000 patients and two concept types.

### 5. Build the provenance graph

The BERT and LLM scripts can build provenance graphs directly when their graph options are enabled. The graph infrastructure supports:

- Entity resolution.
- Never-overwrite temporal versioning.
- Explicit abstained edges.
- Contradiction and disagreement flagging.
- Evidence and gate-diagnostic retention.
- Temporal validity windows.
- Point-in-time querying.
- Aggregate temporal-motif statistics.
- Federation-ready summary aggregation.

The provenance graph is separate from the ground-truth compression graph. The compression graph evaluates known-correct facts; the provenance graph stores model predictions, which may be wrong or abstained.

### 6. Compare model results

After both model runs finish, use the comparison script:

```bash
python compare_bert_llm_results.py \
  --bert_results output/poc_results_clinicalbert_v6.json \
  --llm_results output/poc_results_llm_v7.json
```

Use the actual argument names exposed by the version of the script in the repository.

The comparison should verify:

- Same corpus directory.
- Same seed.
- Compatible patient-level split.
- Different test-set sizes are reported transparently.
- Certification settings are not incorrectly treated as directly comparable.


## Citation and intended use

This repository is intended for research reproducibility and mechanism validation. It should not be used for clinical decision-making or deployment without:

- Validation on real clinical text.
- Expert review of temporal relations.
- Manual evidence-span annotation.
- Evaluation of calibration and abstention behavior.
- Privacy, security, and regulatory review.
- Independent replication.

