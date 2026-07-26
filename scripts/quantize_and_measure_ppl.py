"""
PRISM — Nivel 1, paso 2: cuantizacion real + perplejidad.

Mide Delta-perplejidad (fp32 -> int8 fake-quant) para el modelo con SVMO
fusionado vs. el modelo con LoRA fusionado, sobre el MISMO held-out de
Alpaca que usa el resto del laboratorio (_held_out_batches en run_s3_train.py:
ultimos 8 ejemplos del split train, mismo template de prompt).

Streaming layer-major (reutiliza streaming_loader.py, ya validado en 7B real
por USF/Thm4): el modelo NUNCA esta resuelto entero en RAM. Una capa a la
vez, liberada al terminar.

CORRECCION (2026-07-25) tras un OOM real matando el proceso a las 3/28 capas
con las 4 configuraciones (svmo_fp32, svmo_q8, lora_fp32, lora_q8) resueltas
en paralelo en un runner de 6GB de RAM disponibles: el script ahora procesa
SOLO DOS configuraciones por invocacion -- el par (fp32, q8) de UNA variante
(svmo o lora) -- elegido por --variant. Correr las dos variantes como dos
procesos SEPARADOS Y SECUENCIALES (no simultaneos: correrlos a la vez
duplicaria el mismo pico de RAM que ya causo el OOM). Cada proceso carga
solo el checkpoint que necesita, y termina/libera toda su memoria al SO
antes de que arranque el otro -- sin fragmentacion acumulada entre variantes.
Ademas: gc.collect() explicito por capa (no confiar en la recoleccion lazy
de Python bajo presion de memoria).

"Fake quantization": cuantiza y de-cuantiza los pesos (redondeo simetrico
por canal de salida) pero computa en fp32 -- mide el error de cuantizacion
sin necesitar kernels de bajo-bit ni GPU.

Uso:
    ./.venv/bin/python quantize_and_measure_ppl.py --variant svmo
    ./.venv/bin/python quantize_and_measure_ppl.py --variant lora
(uno despues del otro, nunca los dos a la vez)
"""

import argparse
import gc
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/gula/Documentos/Proyectos Github/lowrank-field-adapters")
from src.training.streaming_loader import (
    ShardReader, build_streaming_model, materialize_shared, _set_weight, find_snapshot,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_merge_outliers import (
    S3_CKPT_PATH, LORA_CKPT_PATH, MATRICES, N_LAYERS,
    load_svd_factors, svmo_delta, lora_delta,
)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = torch.device("cpu")
N_HELDOUT = 16
MAX_SEQ = 384
QUANT_BITS = 4

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)


def fake_quantize(W: torch.Tensor, bits: int = 8) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    scale = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    q = torch.clamp(torch.round(W / scale), -qmax - 1, qmax)
    return q * scale


def held_out_batches(tok):
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.select(range(len(ds) - N_HELDOUT, len(ds)))
    out = []
    for ex in ds:
        text = f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex.get('output','')}"
        ids = tok.encode(text, truncation=True, max_length=MAX_SEQ)
        if len(ids) > 2:
            out.append(torch.tensor(ids).unsqueeze(0))
    return out


def merged_matrix(W, hf_sub, dir_name, layer_idx, ckpt_state, variant, quantize):
    if variant == "svmo":
        U_k, S_k, Vt_k = load_svd_factors(layer_idx, dir_name)
        mlp_prefix = f"model.layers.{layer_idx}.{hf_sub}.modulation."
        mlp_state = {k[len(mlp_prefix):]: v for k, v in ckpt_state.items() if k.startswith(mlp_prefix)}
        delta = svmo_delta(S_k, U_k, Vt_k, mlp_state)
    else:
        A = ckpt_state[f"model.layers.{layer_idx}.{hf_sub}.lora_A"]
        B = ckpt_state[f"model.layers.{layer_idx}.{hf_sub}.lora_B"]
        delta = lora_delta(A, B)
    merged = (W + delta).to(torch.float32)
    if quantize:
        merged = fake_quantize(merged, bits=QUANT_BITS)
    return merged


@torch.no_grad()
def run_variant(variant: str, n_layers: int = N_LAYERS):
    from transformers import AutoTokenizer

    ckpt_path = S3_CKPT_PATH if variant == "svmo" else LORA_CKPT_PATH
    print(f"[PRISM] variante={variant}  cargando tokenizer + checkpoint ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    ckpt_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    seqs = held_out_batches(tok)
    print(f"[PRISM] held-out: {len(seqs)} secuencias, largos {[s.shape[1] for s in seqs]}", flush=True)

    snapshot_dir = find_snapshot(MODEL_NAME)
    reader = ShardReader(snapshot_dir)

    print("[PRISM] construyendo esqueleto del modelo (meta device) ...", flush=True)
    model, cfg = build_streaming_model(MODEL_NAME)
    materialize_shared(model, reader, embed_device=DEVICE, head_device=DEVICE, norm_device=DEVICE)
    # CORRECCION (2026-07-25): embed_tokens y lm_head son ~545M params cada uno
    # (vocab 152064 x hidden 3584, sin tie_word_embeddings). Castear ambos a
    # fp32 duplicaba ~2.2GB de RAM que nunca hacia falta -- ninguno de los dos
    # participa en el merge SVMO/LoRA. Se dejan en su dtype nativo (bf16);
    # solo se sube a fp32 la salida chica de cada uno (embedding de la
    # secuencia, logits) donde realmente hace falta precision.
    _set_weight(model.model.norm, "weight", model.model.norm.weight.float())
    layers = model.model.layers
    rotary = model.model.rotary_emb
    embed_dtype = model.model.embed_tokens.weight.dtype
    head_dtype = model.lm_head.weight.dtype

    configs = [f"{variant}_fp32", f"{variant}_q{QUANT_BITS}"]
    H = {c: [model.model.embed_tokens(s.to(DEVICE)).float() for s in seqs] for c in configs}

    def causal_mask(S, dtype):
        m = torch.full((S, S), float("-inf"), dtype=dtype)
        return torch.triu(m, diagonal=1).view(1, 1, S, S)

    masks = [causal_mask(s.shape[1], torch.float32) for s in seqs]
    pos_embs = [rotary(H[configs[0]][j], torch.arange(seqs[j].shape[1]).unsqueeze(0))
                for j in range(len(seqs))]

    t0 = time.time()
    for layer_idx in range(n_layers):
        blk = layers[layer_idx]
        pre = f"model.layers.{layer_idx}."
        _set_weight(blk.input_layernorm, "weight", reader.get(f"{pre}input_layernorm.weight").float().to(DEVICE))
        _set_weight(blk.post_attention_layernorm, "weight",
                    reader.get(f"{pre}post_attention_layernorm.weight").float().to(DEVICE))

        base_W, base_bias = {}, {}
        for hf_sub, dir_name in MATRICES:
            base_W[dir_name] = reader.get(f"{pre}{hf_sub}.weight").to(DEVICE).float()
            bname = f"{pre}{hf_sub}.bias"
            base_bias[dir_name] = reader.get(bname).to(DEVICE).float() if reader.has(bname) else None

        for cfg_name in configs:
            quantize = cfg_name.endswith(f"q{QUANT_BITS}")
            for hf_sub, dir_name in MATRICES:
                sub, proj = hf_sub.split(".")
                lin = getattr(getattr(blk, sub), proj)
                W_merged = merged_matrix(base_W[dir_name], hf_sub, dir_name, layer_idx,
                                          ckpt_state, variant, quantize)
                _set_weight(lin, "weight", W_merged)
                if base_bias[dir_name] is not None:
                    _set_weight(lin, "bias", base_bias[dir_name])

            for j in range(len(seqs)):
                out = blk(H[cfg_name][j], attention_mask=masks[j], position_embeddings=pos_embs[j])
                H[cfg_name][j] = out[0] if isinstance(out, tuple) else out

        # CORRECCION (2026-07-25): liberar los pesos de ESTA capa antes de pasar
        # a la siguiente. Sin esto, cada capa procesada se queda resuelta en RAM
        # para siempre (nunca se vuelve a tocar ese modulo), acumulando ~230M
        # parametros/capa hasta reventar por OOM -- exactamente lo que paso.
        # Replica FrozenStreamer.unload() de streaming_loader.py.
        meta = torch.device("meta")
        for hf_sub, dir_name in MATRICES:
            sub, proj = hf_sub.split(".")
            lin = getattr(getattr(blk, sub), proj)
            for attr in ("weight", "bias"):
                p = getattr(lin, attr, None)
                if isinstance(p, torch.Tensor):
                    setattr(lin, attr, torch.nn.Parameter(p.data.to(meta), requires_grad=False))
        for norm_name in ("input_layernorm", "post_attention_layernorm"):
            m = getattr(blk, norm_name)
            m.weight = torch.nn.Parameter(m.weight.data.to(meta), requires_grad=False)

        del base_W, base_bias
        gc.collect()
        print(f"[PRISM] {variant}: capa {layer_idx+1}/{n_layers} lista ({time.time()-t0:.0f}s)", flush=True)

    print("[PRISM] norma final + lm_head + perdida ...", flush=True)
    results = {}
    for cfg_name in configs:
        total_loss, total_tok = 0.0, 0
        for j, s in enumerate(seqs):
            h = model.model.norm(H[cfg_name][j])
            logits = model.lm_head(h.to(head_dtype)).float()
            labels = s.to(DEVICE)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tok += shift_labels.numel()
        ppl = float(torch.exp(torch.tensor(total_loss / total_tok)))
        results[cfg_name] = {"perplexity": ppl, "total_tokens": total_tok}
        print(f"[PRISM] {cfg_name}: ppl={ppl:.4f}  tokens={total_tok}", flush=True)

    results["_meta"] = {
        "variant": variant, "model": MODEL_NAME, "n_layers": n_layers, "quant_bits": QUANT_BITS,
        "n_heldout": len(seqs), "max_seq": MAX_SEQ, "elapsed_seconds": time.time() - t0,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"quant_ppl_{variant}_{QUANT_BITS}bit.json")
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"[PRISM] listo -> {out_path}", flush=True)

    d = results[f"{variant}_q{QUANT_BITS}"]["perplexity"] - results[f"{variant}_fp32"]["perplexity"]
    print(f"\nDelta-ppl {variant} (fp32->int{QUANT_BITS}): {d:+.4f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=["svmo", "lora"])
    p.add_argument("--n_layers", type=int, default=N_LAYERS)
    p.add_argument("--bits", type=int, default=QUANT_BITS)
    args = p.parse_args()
    QUANT_BITS = args.bits
    run_variant(args.variant, args.n_layers)
