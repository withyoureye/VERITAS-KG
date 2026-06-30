#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs data/processed results

COMMON_SEEDS=(--seed 0 --seed 1 --seed 2 --seed 3 --seed 4)

launch() {
  local name="$1"
  local gpu="$2"
  shift 2
  local log_path="logs/${name}.log"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" "$@" > "$log_path" 2>&1 &
  local pid="$!"
  echo "$pid" > "logs/${name}.pid"
  echo "${name}: pid=${pid}, gpu=${gpu}, log=${log_path}"
}

launch icews14_real_full 0 bash -lc '
  set -euo pipefail
  conda run -n rcke python scripts/prepare_icews14_real.py \
    --data-root ../data \
    --output-dir data/processed/icews14_real_full \
    --train-limit -1 \
    --test-limit -1 \
    --negative-candidates 5 \
    --seed 0
  conda run -n rcke python experiments/run_experiment.py \
    --input data/processed/icews14_real_full/assertions.jsonl \
    --output results/icews14_real_full \
    --run-name icews14_real_full \
    --seed 0 --seed 1 --seed 2 --seed 3 --seed 4 \
    --seed-jitter 0.03
'

launch fever_real_full 1 bash -lc '
  set -euo pipefail
  conda run -n rcke python scripts/prepare_fever_real.py \
    --data-root ../data \
    --output-dir data/processed/fever_real_full \
    --split dev \
    --limit -1
  conda run -n rcke python experiments/run_experiment.py \
    --input data/processed/fever_real_full/assertions.jsonl \
    --output results/fever_real_full \
    --run-name fever_real_full \
    --seed 0 --seed 1 --seed 2 --seed 3 --seed 4 \
    --seed-jitter 0.02
'

launch yago_type_reasoning_full 2 bash -lc '
  set -euo pipefail
  conda run -n rcke python scripts/prepare_yago_type_reasoning.py \
    --data-root ../data \
    --output-dir data/processed/yago_type_reasoning_full \
    --train-limit -1 \
    --valid-limit -1 \
    --test-limit -1
  conda run -n rcke python experiments/run_experiment.py \
    --input data/processed/yago_type_reasoning_full/assertions.jsonl \
    --output results/yago_type_reasoning_full \
    --run-name yago_type_reasoning_full \
    --seed 0 --seed 1 --seed 2 --seed 3 --seed 4 \
    --seed-jitter 0.02
'
