# Reviewer Reproduction Guide

This bundle contains the source code and experiment scripts for the VERITAS-KG experiments. It is designed for a lightweight reviewer check first, followed by the paper-facing FEVER experiments.

## 1. Directory Layout

Assume the bundle is located at:

```bash
export PROJECT_ROOT=/path/to/VERITAS-KG
```

All commands below are run from:

```bash
cd "$PROJECT_ROOT"
```

Raw datasets are expected under:

```bash
export DATA_ROOT=/path/to/data
```

From this bundle, the default relative data path is:

```bash
../data
```

The bundle intentionally excludes large raw datasets, model weights, cached predictions, and result folders. If a command depends on cached outputs, this guide marks it explicitly.

## 2. Environment

Recommended environment:

```bash
conda activate rcke
```

Or run commands without activating:

```bash
conda run -n rcke python <script.py>
```

Use a single visible GPU only:

```bash
export CUDA_VISIBLE_DEVICES=0
```

Do not use multi-GPU execution; the server does not have NVLink. Most symbolic KG steps are CPU-bound, so low GPU utilization is expected unless running neural verifiers or Qwen readers.

Minimum Python packages used by the scripts include:

```text
numpy
scikit-learn
torch
transformers
sentencepiece
modelscope
rdflib
pyshacl
owlready2
matplotlib
```

If a package is missing, install it inside `rcke`.

## 3. Data Expected

The scripts expect these local files when running the corresponding experiments:

```text
../data/ICEWS14/train.txt
../data/ICEWS14/test.txt
../data/fever/paper_dev.jsonl
../data/fever/paper_test.jsonl
../data/fever/wiki-pages.zip
../data/downloads/scifact_official/data/claims_dev.jsonl
../data/downloads/scifact_official/data/corpus.jsonl
../data/YAGO/...
```

`valid.txt` for ICEWS14 is optional. The ICEWS14 preparation script continues if it is absent.

## 4. Quick Smoke Test

Run this first to verify that the symbolic KG pipeline works:

```bash
cd "$PROJECT_ROOT"

conda run -n rcke python scripts/prepare_samples.py \
  --data-root ../data \
  --output-dir data/processed/smoke \
  --icews-limit 12 \
  --fever-limit 6 \
  --yago-limit 8 \
  --inject-conflicts 4 \
  --inject-invalid 3

conda run -n rcke python experiments/run_experiment.py \
  --input data/processed/smoke/assertions.jsonl \
  --output results/smoke \
  --run-name smoke
```

Expected outputs:

```text
results/smoke/metrics.json
results/smoke/summary.md
results/smoke/ablation.csv
```

## 5. Main FEVER Test-Style Experiment

This is the main paper-facing setting. It uses `paper_dev` for calibration and `paper_test` for frozen evaluation. Do not tune any parameter on `paper_test`.

Build the local FEVER wiki index if it is not already present:

```bash
cd "$PROJECT_ROOT"

conda run -n rcke python scripts/build_fever_wiki_index.py \
  --fever-root ../data/fever \
  --output data/processed/fever_wiki_index_dev_test.json
```

Prepare FEVER test-style assertions:

```bash
conda run -n rcke python scripts/prepare_fever_real.py \
  --data-root ../data \
  --output-dir data/processed/fever_test_teststyle_v4_wiki \
  --split test \
  --limit -1 \
  --wiki-index data/processed/fever_wiki_index_dev_test.json
```

Run KG scorer:

```bash
conda run -n rcke python experiments/run_experiment.py \
  --input data/processed/fever_test_teststyle_v4_wiki/assertions.jsonl \
  --output results/fever_test_teststyle_kg \
  --run-name fever_test_teststyle_kg \
  --seed 0 \
  --seed-jitter 0 \
  --evidence-scorer auto
```

Run the FEVER dev-to-test-style neural verifier and fusion protocol:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n rcke python scripts/run_fever_teststyle_protocol.py \
  --output results/fever_dev_to_teststyle \
  --batch-size 32 \
  --bootstrap-samples 5000
```

Expected key outputs:

```text
results/fever_dev_to_teststyle/metrics.json
results/fever_dev_to_teststyle/summary.md
results/fever_dev_to_teststyle/predictions.jsonl
```

Main reported result from the completed run:

```text
FEVER/ANLI DeBERTa verifier only: accuracy 0.7361, macro-F1 0.7283
KG scorer only:                  accuracy 0.6988, macro-F1 0.6272
KG + verifier fusion:            accuracy 0.9487, macro-F1 0.9486
```

## 6. KG-Enhanced Baselines And FEVER Ablations

This experiment compares against structured/KG-enhanced baselines and runs FEVER main-task ablations.

It reuses cached outputs from:

```text
results/fever_test_teststyle_kg/
results/fever_dev_to_teststyle/
```

Run:

```bash
cd "$PROJECT_ROOT"

CUDA_VISIBLE_DEVICES=0 conda run -n rcke python scripts/run_fever_kg_baselines_and_ablations.py \
  --output results/fever_kg_baselines_ablations \
  --bootstrap-samples 5000
```

Expected outputs:

```text
results/fever_kg_baselines_ablations/metrics.json
results/fever_kg_baselines_ablations/summary.md
results/fever_kg_baselines_ablations/predictions.jsonl
```

Completed-run headline results:

```text
Structured evidence fusion baseline: accuracy 0.6807, macro-F1 0.6281
Graph-prior text fusion baseline:    accuracy 0.6877, macro-F1 0.6417
KG scorer only:                      accuracy 0.6988, macro-F1 0.6272
VERITAS-KG full fusion:              accuracy 0.9487, macro-F1 0.9486
No evidence reliability:             accuracy 0.9406, macro-F1 0.9401
```

Ontology and context ablations are intentionally non-discriminative on FEVER because FEVER candidate assertions share the same `Claim-has_verdict-Verdict` signature and do not contain temporal/location context.

## 7. Strong Verifier Baseline

The stronger neural verifier baseline uses a DeBERTa-v3-large FEVER/ANLI model. If the model is absent, download it through Hugging Face or ModelScope in the `rcke` environment.

Representative command:

```bash
cd "$PROJECT_ROOT"

CUDA_VISIBLE_DEVICES=0 conda run -n rcke python scripts/run_fever_strong_verifier_sweep.py \
  --output results/fever_fever_tuned_verifier_sweep \
  --model MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli \
  --batch-size 32 \
  --bootstrap-samples 5000
```

Completed-run headline result:

```text
DeBERTa-v3-large FEVER/ANLI verifier only: accuracy 0.7514, macro-F1 0.7443
KG + FEVER/ANLI verifier fusion:           accuracy 0.9560, macro-F1 0.9561
Selected alpha: 0.9
Calibration split: 2,019 claims
Held-out split: 7,980 claims
```

## 8. SciFact Transfer Bottleneck Diagnostic

This diagnostic quantifies why FEVER-calibrated fusion transfers weakly to document-level SciFact evidence and improves when sentence-level scientific provenance is available.

It reuses cached SciFact verifier outputs from:

```text
results/cross_dataset_transfer/
```

Run:

```bash
cd "$PROJECT_ROOT"

CUDA_VISIBLE_DEVICES=0 conda run -n rcke python scripts/run_scifact_transfer_diagnostics.py \
  --output results/scifact_transfer_diagnostics \
  --bootstrap-samples 5000
```

Expected outputs:

```text
results/scifact_transfer_diagnostics/metrics.json
results/scifact_transfer_diagnostics/quality_metrics.csv
results/scifact_transfer_diagnostics/summary.md
results/scifact_transfer_diagnostics/predictions.jsonl
```

Completed-run headline result:

```text
FEVER effective structured evidence rate:        0.5655, fusion gain +0.2126
SciFact cited-doc effective structured rate:     0.0000, fusion gain +0.0067
SciFact gold-rationale effective structured rate:0.4167, fusion gain +0.3733
```

The local SciFact corpus does not contain journal rank or citation count. The current domain adaptation uses available provenance buckets only: cited document, rationale sentence, structured abstract flag if present, and lexical claim-evidence overlap.

## 9. ICEWS14 And YAGO Mechanism Checks

ICEWS14 and YAGO are not the main FEVER reliability benchmark. They are mechanism analyses.

ICEWS14 temporal perturbation diagnostic:

```bash
cd "$PROJECT_ROOT"

CUDA_VISIBLE_DEVICES=0 conda run -n rcke python scripts/run_icews_temporal_perturbation.py \
  --input data/processed/icews14_real_full/assertions.jsonl \
  --output results/icews14_temporal_perturbation \
  --groups 5000 \
  --seed 0 \
  --shift-days 30 \
  --hard-shift-days 1
```

YAGO import-validation baselines:

```bash
cd "$PROJECT_ROOT"

conda run -n rcke python scripts/run_yago_import_validation_baselines.py \
  --input data/processed/yago_type_reasoning_full/assertions.jsonl \
  --output results/yago_import_validation_baselines
```

## 10. Where To Read Results

Primary summaries:

```text
PAPER_TABLES.md
FEVER_STRONG_BASELINE_SUMMARY.md
SCIFACT_TRANSFER_SUMMARY.md
RESULTS_SUMMARY.md
README_experiments.md
```

Primary result folders in the full working repository:

```text
results/fever_dev_to_teststyle/
results/fever_kg_baselines_ablations/
results/fever_fever_tuned_verifier_sweep/
results/scifact_transfer_diagnostics/
results/cross_dataset_transfer/
results/yago_import_validation_baselines/
results/icews14_temporal_perturbation/
```

## 11. Common Failure Modes

- `FileNotFoundError` for data: verify `../data` points to the dataset directory, or pass explicit `--data-root "$DATA_ROOT"`.
- Missing FEVER wiki index: run `scripts/build_fever_wiki_index.py`.
- Missing DeBERTa/Qwen weights: download them inside `rcke` or adjust `--model` to a local path.
- CUDA out of memory: reduce `--batch-size`; keep `CUDA_VISIBLE_DEVICES=0`.
- Low GPU utilization: expected for symbolic KG, preprocessing, bootstrap, and cached-result diagnostics.
- SciFact citation/journal metadata missing: expected for the local SciFact files; current adapter reports this limitation explicitly.

## 12. Minimal Reviewer Checklist

For a fast reproducibility audit, run:

```bash
conda run -n rcke python scripts/prepare_samples.py --data-root ../data --output-dir data/processed/smoke --icews-limit 12 --fever-limit 6 --yago-limit 8 --inject-conflicts 4 --inject-invalid 3
conda run -n rcke python experiments/run_experiment.py --input data/processed/smoke/assertions.jsonl --output results/smoke --run-name smoke
```

Then inspect:

```text
results/smoke/summary.md
results/smoke/metrics.json
PAPER_TABLES.md
FEVER_STRONG_BASELINE_SUMMARY.md
```

For the full paper-facing FEVER claim, run Sections 5 to 8 in order.
