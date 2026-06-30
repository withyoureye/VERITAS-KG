#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

for name in icews14_real_full fever_real_full yago_type_reasoning_full; do
  echo "### ${name}"
  pid_file="logs/${name}.pid"
  log_file="logs/${name}.log"
  if [[ -f "$pid_file" ]]; then
    pid="$(cat "$pid_file")"
    if ps -p "$pid" > /dev/null 2>&1; then
      echo "status: running pid=${pid}"
      ps -o pid,ppid,stat,pcpu,pmem,etime,rss,cmd -p "$pid"
      pstree -ap "$pid" | tail -8 || true
    else
      echo "status: finished pid=${pid}"
    fi
  else
    echo "status: no pid file"
  fi

  if [[ -f "$log_file" ]]; then
    echo "log: ${log_file}"
    tail -20 "$log_file"
  else
    echo "log: missing"
  fi

  processed_dir="data/processed/${name}"
  results_dir="results/${name}"
  [[ -e "$processed_dir" ]] && du -sh "$processed_dir"
  [[ -e "$results_dir" ]] && du -sh "$results_dir"
  echo
done
