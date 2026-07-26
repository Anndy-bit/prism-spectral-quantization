#!/bin/bash
# PRISM -- corre las 5 mediciones (base, svmo@8, lora@8, svmo@4, lora@4)
# con held-out de 16 ejemplos, una despues de la otra (nunca en paralelo,
# para no repetir el OOM). Cada fase escribe su propio log.
set -e
cd "/home/gula/Documentos/Proyectos Github/lowrank-field-adapters"
export HF_HUB_OFFLINE=1
export DNNL_MAX_CPU_ISA=SSE41
PY="./.venv/bin/python -u"
SCRIPTS="/home/gula/Documentos/Proyectos Github/prism-spectral-quantization/scripts"
LOGDIR="/tmp/claude-1000/-home-gula-Documentos-Proyectos-Github-lowrank-field-adapters/903034aa-9843-4426-896e-1393babe955f/scratchpad"

run() {
    local name="$1"; shift
    echo "=== [$(date '+%H:%M:%S')] arrancando: $name ===" | tee -a "$LOGDIR/prism_16ex_ALL.log"
    $PY -c "
import torch
torch.backends.mkldnn.enabled = False
import sys
sys.path.insert(0, '$SCRIPTS')
$*
" >> "$LOGDIR/prism_16ex_${name}.log" 2>&1
    echo "=== [$(date '+%H:%M:%S')] termino: $name ===" | tee -a "$LOGDIR/prism_16ex_ALL.log"
}

run base    "import measure_base_ppl as m; m.main()"
run svmo8   "import quantize_and_measure_ppl as q; q.QUANT_BITS = 8; q.run_variant('svmo', 28)"
run lora8   "import quantize_and_measure_ppl as q; q.QUANT_BITS = 8; q.run_variant('lora', 28)"
run svmo4   "import quantize_and_measure_ppl as q; q.QUANT_BITS = 4; q.run_variant('svmo', 28)"
run lora4   "import quantize_and_measure_ppl as q; q.QUANT_BITS = 4; q.run_variant('lora', 28)"

echo "=== [$(date '+%H:%M:%S')] TODO TERMINADO ===" | tee -a "$LOGDIR/prism_16ex_ALL.log"
