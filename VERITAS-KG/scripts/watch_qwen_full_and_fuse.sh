#!/usr/bin/env bash
set -euo pipefail

PID_FILE="${1:-logs/qwen_fever_evidence_reader_full.pid}"
QWEN_JSON="${2:-results/baselines_plus/qwen_fever_evidence_reader_full/qwen_fever_evidence_reader.json}"
OUTPUT_DIR="${3:-results/baselines_plus/fever_kg_qwen_fusion_full}"

if [[ ! -s "$PID_FILE" ]]; then
  echo "Missing pid file: $PID_FILE" >&2
  exit 1
fi

pid="$(cat "$PID_FILE")"
while kill -0 "$pid" >/dev/null 2>&1; do
  sleep 60
done

if [[ ! -s "$QWEN_JSON" ]]; then
  echo "Qwen output missing after process exit: $QWEN_JSON" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=1 conda run -n rcke python scripts/fuse_fever_kg_qwen.py \
  --kg-rankings results/fever_real_full_v4_wiki_auto/seed_0/full/rankings.jsonl \
  --qwen "$QWEN_JSON" \
  --output "$OUTPUT_DIR" \
  --alpha 0.0 --alpha 0.25 --alpha 0.5 --alpha 0.75 --alpha 0.9 --alpha 1.0 \
  --qwen-boost 1.0

CUDA_VISIBLE_DEVICES=1 conda run -n rcke python scripts/write_paper_tables.py \
  --root . \
  --output PAPER_TABLES.md
