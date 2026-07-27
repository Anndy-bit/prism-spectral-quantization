# PRISM: Precision Robustness in Spectral vs. Additive Modulation

PRISM compares two parameter-efficient fine-tuning (PEFT) adapters, LoRA and
SVMO, along a path the quantization literature rarely studies directly:
train, merge the adapter into the frozen weights, then quantize the merged
result for deployment. Most prior work quantizes the base model first and
trains an adapter on top of it; this project asks what happens when the
order is reversed, and whether an adapter's update geometry, not just its
size, predicts how well the merge survives quantization.

The full methodology, related work, and results are in
[`paper/main.tex`](paper/main.tex). This README documents the repository;
see the paper for the argument.

## Result summary

Ranking the two adapters by the raw magnitude of their merged weight update
suggests SVMO's smaller, spectrally-bounded update is the safer one to
quantize. A magnitude-matched control, which rescales LoRA's update to
SVMO's own size before comparing, reverses that ranking. Measured 4-bit
post-training quantization confirms the control's prediction rather than
the raw comparison's: SVMO's merged model degrades more than LoRA's,
despite its update being 47x smaller in Frobenius norm. Full numbers are in
[`docs/experimental-results.md`](docs/experimental-results.md).

## Repository structure

```
paper/        LaTeX source and compiled PDF of the paper
src/prism/    Library code: config, adapter merge math, metrics,
              checkpoint I/O, and the shared streaming forward pass
scripts/      Thin CLI entry points that call into src/prism
results/      CSV and JSON outputs from the scripts (checked in)
docs/         Experimental methodology and results, in more detail
              than the paper
```

## Reproducing the results

The scripts expect a checked-out copy of
[`lowrank-field-adapters`](https://github.com/Anndy-bit/lowrank-field-adapters)
(for the trained SVMO/LoRA checkpoints and the streaming model loader) and a
cached local copy of `Qwen/Qwen2.5-7B-Instruct`. Paths default to the layout
used to produce the paper's results and can be overridden with environment
variables documented in `src/prism/config.py` (`PRISM_LFA_REPO`,
`PRISM_SVD_DIR`, `PRISM_S3_CKPT`, `PRISM_LORA_CKPT`, `PRISM_HF_HUB_DIR`).

```bash
# weight-distribution statistics (Section III of the paper)
python scripts/analyze_merge_outliers.py
python scripts/analyze_magnitude_matched.py

# held-out perplexity, base model and both adapters at 8-bit and 4-bit
python scripts/measure_base_ppl.py
python scripts/quantize_and_measure_ppl.py --variant svmo --bits 4
python scripts/quantize_and_measure_ppl.py --variant lora --bits 4

# or run the perplexity measurements in sequence, one process at a time
scripts/run_experiments.sh
```

Every script streams the model one decoder layer at a time and runs
entirely on CPU; no GPU is required or used. On the hardware this project
was developed on, each perplexity measurement takes roughly an hour.

## Relationship to `lowrank-field-adapters`

PRISM is a companion project to `lowrank-field-adapters`, which introduces
the S³ adapter (of which SVMO is one component) and reports its downstream
task accuracy. PRISM reuses the checkpoints trained there but asks an
independent question, about post-training quantization robustness rather
than task accuracy, and is self-contained: this repository does not require
retraining anything.
