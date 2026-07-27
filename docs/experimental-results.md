# Experimental Results

This document records the methodology and results behind `paper/main.tex` in
more detail than the paper itself, for reproducibility. All numbers below
match the paper; where the paper reports an aggregate, this document also
gives the per-matrix and per-layer breakdown.

Model: Qwen2.5-7B-Instruct (28 layers, hidden size 3584). Adapters: SVMO
(rank `k=128`, modulation width `H=32`) and LoRA (`r=8`, `alpha=16`), both
from the same training run used throughout the `lowrank-field-adapters`
project. Held-out set: the last 16 instruction-response pairs of the Alpaca
training split, 1,978 scored tokens total, evaluated via CPU layer-major
streaming (one decoder layer materialized at a time).

## Stage 1: Raw Comparison

Script: `scripts/analyze_merge_outliers.py`. Output: `results/outlier_stats.csv`
(588 rows: 28 layers x 7 matrices x 3 variants).

For every adapted matrix, computes excess kurtosis and the max-to-standard-
deviation ratio of the base weight, the SVMO-merged weight, and the
LoRA-merged weight, plus each merge's Frobenius norm relative to the base
weight's.

Averaged over all 196 matrices:

| | Base | +SVMO | +LoRA |
|---|---:|---:|---:|
| Kurtosis | 5.6745 | 5.6745 | 5.5853 |
| Max/std | 4.8799 | 4.8799 | 4.8776 |
| Frobenius norm / \|W\| | -- | 0.051% | 2.400% |

SVMO's update averages 47x smaller than LoRA's in relative Frobenius norm
and leaves both weight-distribution statistics essentially unchanged. LoRA's
merge shifts both, concentrated in the MLP projections: `gate_proj`'s mean
absolute kurtosis shift is 1,104x larger for LoRA than SVMO; `up_proj`'s is
390x larger.

Taken at face value, this says SVMO's merge is the safer one to quantize:
smaller in magnitude and gentler on the statistics per-channel quantizers
depend on. This comparison is confounded by update size and is not the
paper's conclusion; see Stage 2.

### Depth dependence

Splitting the 196 matrices into layers 0-13 and layers 14-27 shows the raw
gap is not uniform by depth: the mean absolute kurtosis shift is 627x larger
for LoRA than SVMO in the early half, against 7.9x in the late half. Nearly
all of the early-layer gap traces to `gate_proj`, whose own baseline
kurtosis is 709.6 at layer 0, 139.5 at layer 1, and 67.1 at layer 2 --
one to three orders of magnitude larger than every other matrix at those
depths, and the same weight-outlier concentration that motivates AWQ,
SpinQuant, and GPTQ's calibrated quantizers.

## Stage 2: Magnitude-Matched Control

Script: `scripts/analyze_magnitude_matched.py`. Output:
`results/magnitude_matched_stats.csv` (196 rows).

Rescales LoRA's merge delta, per matrix, to exactly match SVMO's Frobenius
norm (isotropic scaling: direction preserved, magnitude matched), then
repeats the Stage 1 measurement. This isolates the effect of update
*geometry* from update *size*.

At matched magnitude, the ranking reverses: LoRA disturbs kurtosis less than
SVMO in 188 of 196 matrices, and the max-to-standard-deviation ratio less in
179 of 196. The two projections that showed LoRA as over 1,000x more
disruptive in the raw comparison (`gate_proj`, `up_proj`) show SVMO as the
more disruptive one instead once magnitude is matched. Within the extreme
early layers identified in Stage 1 (layers 0-2, 21 matrices), the same
reversal holds in 20 of 21.

This reversal is a hypothesis about what happens under real quantization,
not a measurement of it; Stage 3 is the check.

## Stage 3: Real Post-Training Quantization

Scripts: `scripts/measure_base_ppl.py`, `scripts/quantize_and_measure_ppl.py`.
Outputs: `results/base_ppl.json`, `results/quant_ppl_{svmo,lora}_{8,4}bit.json`.

Each merged checkpoint is quantized with a symmetric per-output-channel
quantize-dequantize round-trip (8-bit and 4-bit), and held-out perplexity is
measured before and after, on the same 16-example set used in Stages 1-2.

| | fp32 | 8-bit | 4-bit |
|---|---:|---:|---:|
| Base (no adapter) | 7.05 | -- | -- |
| +SVMO | 7.02 | 7.04 (+0.021) | 11.77 (+4.751) |
| +LoRA | 4.82 | 4.82 (+0.002) | 8.53 (+3.712) |

At 8-bit, both changes are near zero. At 4-bit, SVMO's merged model degrades
more (+4.751 perplexity) than LoRA's (+3.712), the same order the Stage 2
control predicted and the opposite order the Stage 1 raw comparison would
have predicted.

## Summary

The three stages, run in this order, tell a consistent story: the raw
comparison (Stage 1) favors SVMO and is confounded by update size; the
magnitude-matched control (Stage 2) reverses that ranking once size is
controlled for; the real quantization measurement (Stage 3) confirms the
control's prediction rather than the raw comparison's. Update size and
update robustness to quantization are not the same property.

See `paper/main.tex`, Sections III-IV, for the full methodology, related
work, and discussion of limitations (single training run, single base
model, simulated rather than deployed quantization).
