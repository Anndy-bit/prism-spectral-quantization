"""Layer-major streaming forward pass, shared by every perplexity measurement.

Materializes one decoder layer's weights at a time from disk, runs every
held-out sequence through it for every requested configuration, then frees
that layer back to the ``meta`` device before moving to the next one. Peak
memory never holds more than one layer's weights regardless of model size,
which is what makes evaluating a 7B model on a CPU with a few GB of RAM
possible.

A single loop serves both "no adapter" and "adapter merged and quantized"
measurements: what differs between them is only the per-matrix weight each
config supplies, via ``weight_fn``.
"""

import sys
import time
from typing import Callable, Sequence

import torch
import torch.nn.functional as F

from prism.config import LFA_REPO, MATRICES, MODEL_NAME

sys.path.insert(0, LFA_REPO)
from src.training.streaming_loader import (  # noqa: E402
    ShardReader,
    build_streaming_model,
    materialize_shared,
    _set_weight,
    find_snapshot,
)

DEVICE = torch.device("cpu")

# Given (config name, HF submodule path, short matrix name, layer index, base
# weight), return the weight tensor that configuration should use at that
# layer and matrix. Identity for an unmodified baseline; merge (and
# optionally quantize) for an adapter configuration.
WeightFn = Callable[[str, str, str, int, torch.Tensor], torch.Tensor]


def _causal_mask(seq_len: int, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype)
    return torch.triu(mask, diagonal=1).view(1, 1, seq_len, seq_len)


def _release_layer_to_meta(block) -> None:
    """Free one decoder layer's weights back to the meta device.

    Without this, every processed layer stays materialized for the rest of
    the run and peak memory grows linearly with model depth instead of
    staying bounded by a single layer.
    """
    meta = torch.device("meta")
    for hf_sub, _ in MATRICES:
        sub, proj = hf_sub.split(".")
        lin = getattr(getattr(block, sub), proj)
        for attr in ("weight", "bias"):
            p = getattr(lin, attr, None)
            if isinstance(p, torch.Tensor):
                setattr(lin, attr, torch.nn.Parameter(p.data.to(meta), requires_grad=False))
    for norm_name in ("input_layernorm", "post_attention_layernorm"):
        norm = getattr(block, norm_name)
        norm.weight = torch.nn.Parameter(norm.weight.data.to(meta), requires_grad=False)


def run_layerwise_perplexity(
    sequences: Sequence[torch.Tensor],
    configs: Sequence[str],
    weight_fn: WeightFn,
    n_layers: int,
    on_layer_done: Callable[[int, float], None] | None = None,
) -> dict:
    """Run every configuration through the streamed model and return perplexity.

    Returns ``{config_name: {"perplexity": float, "total_tokens": int}}``.
    """
    snapshot_dir = find_snapshot(MODEL_NAME)
    reader = ShardReader(snapshot_dir)

    model, _ = build_streaming_model(MODEL_NAME)
    materialize_shared(model, reader, embed_device=DEVICE, head_device=DEVICE, norm_device=DEVICE)
    _set_weight(model.model.norm, "weight", model.model.norm.weight.float())

    layers = model.model.layers
    rotary = model.model.rotary_emb
    head_dtype = model.lm_head.weight.dtype

    hidden = {
        cfg: [model.model.embed_tokens(s.to(DEVICE)).float() for s in sequences]
        for cfg in configs
    }
    masks = [_causal_mask(s.shape[1], torch.float32) for s in sequences]
    position_embeddings = [
        rotary(hidden[configs[0]][j], torch.arange(sequences[j].shape[1]).unsqueeze(0))
        for j in range(len(sequences))
    ]

    t0 = time.time()
    for layer_idx in range(n_layers):
        block = layers[layer_idx]
        prefix = f"model.layers.{layer_idx}."
        _set_weight(block.input_layernorm, "weight",
                    reader.get(f"{prefix}input_layernorm.weight").float().to(DEVICE))
        _set_weight(block.post_attention_layernorm, "weight",
                    reader.get(f"{prefix}post_attention_layernorm.weight").float().to(DEVICE))

        base_weights, base_biases = {}, {}
        for hf_sub, dir_name in MATRICES:
            base_weights[dir_name] = reader.get(f"{prefix}{hf_sub}.weight").to(DEVICE).float()
            bias_name = f"{prefix}{hf_sub}.bias"
            base_biases[dir_name] = (
                reader.get(bias_name).to(DEVICE).float() if reader.has(bias_name) else None
            )

        for cfg in configs:
            for hf_sub, dir_name in MATRICES:
                sub, proj = hf_sub.split(".")
                lin = getattr(getattr(block, sub), proj)
                W = weight_fn(cfg, hf_sub, dir_name, layer_idx, base_weights[dir_name])
                _set_weight(lin, "weight", W)
                if base_biases[dir_name] is not None:
                    _set_weight(lin, "bias", base_biases[dir_name])

            for j in range(len(sequences)):
                out = block(hidden[cfg][j], attention_mask=masks[j],
                            position_embeddings=position_embeddings[j])
                hidden[cfg][j] = out[0] if isinstance(out, tuple) else out

        _release_layer_to_meta(block)
        del base_weights, base_biases

        if on_layer_done is not None:
            on_layer_done(layer_idx, time.time() - t0)

    results = {}
    for cfg in configs:
        total_loss, total_tokens = 0.0, 0
        for j, seq in enumerate(sequences):
            h = model.model.norm(hidden[cfg][j])
            logits = model.lm_head(h.to(head_dtype)).float()
            labels = seq.to(DEVICE)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += shift_labels.numel()
        ppl = float(torch.exp(torch.tensor(total_loss / total_tokens)))
        results[cfg] = {"perplexity": ppl, "total_tokens": total_tokens}

    results["_elapsed_seconds"] = time.time() - t0
    return results
