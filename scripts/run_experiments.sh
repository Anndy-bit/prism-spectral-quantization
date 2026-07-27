#!/bin/bash
# Runs the five held-out perplexity measurements (base, svmo@8, lora@8,
# svmo@4, lora@4) sequentially, never in parallel: running two configurations
# at once roughly doubles peak memory, since each needs its own checkpoint
# and streamed layer resident at the same time.
#
# Env vars (all optional):
#   PRISM_REPO_ROOT   repository root (default: parent of this script's dir)
#   PRISM_PYTHON      python interpreter to use (default: python3)
#   PRISM_LOG_DIR     where per-phase logs are written (default: results/logs)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${PRISM_REPO_ROOT:-$(dirname "$SCRIPT_DIR")}"
PYTHON="${PRISM_PYTHON:-python3}"
LOG_DIR="${PRISM_LOG_DIR:-$REPO_ROOT/results/logs}"

mkdir -p "$LOG_DIR"
ALL_LOG="$LOG_DIR/run_all.log"

# Needed on CPUs without AVX2, where PyTorch's oneDNN backend otherwise
# raises SIGILL; harmless to leave set on CPUs that do have AVX2.
export DNNL_MAX_CPU_ISA=SSE41
export PRISM_DISABLE_MKLDNN=1

run() {
    local name="$1"; shift
    echo "=== [$(date '+%H:%M:%S')] starting: $name ===" | tee -a "$ALL_LOG"
    "$PYTHON" -u "$@" >> "$LOG_DIR/$name.log" 2>&1
    echo "=== [$(date '+%H:%M:%S')] finished: $name ===" | tee -a "$ALL_LOG"
}

cd "$REPO_ROOT"
run base   scripts/measure_base_ppl.py
run svmo8  scripts/quantize_and_measure_ppl.py --variant svmo --bits 8
run lora8  scripts/quantize_and_measure_ppl.py --variant lora --bits 8
run svmo4  scripts/quantize_and_measure_ppl.py --variant svmo --bits 4
run lora4  scripts/quantize_and_measure_ppl.py --variant lora --bits 4

echo "=== [$(date '+%H:%M:%S')] all phases complete ===" | tee -a "$ALL_LOG"
