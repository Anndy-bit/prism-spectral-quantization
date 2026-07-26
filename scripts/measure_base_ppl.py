"""
PRISM -- perplejidad del modelo BASE, sin ningun adaptador, en el MISMO
held-out (mismos 8 ejemplos de Alpaca, mismo template) que se usa para
svmo_fp32 y lora_fp32. Necesario para saber si "SVMO solo" y "LoRA" de
verdad llegan a un desempeno comparable, o si SVMO solo esta cerca del
modelo sin adaptar (tal como sugiere ablation_pathways.md para el
checkpoint de ablacion rapida -- hay que confirmarlo para ESTE checkpoint).
"""

import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/gula/Documentos/Proyectos Github/lowrank-field-adapters")
from src.training.streaming_loader import (
    ShardReader, build_streaming_model, materialize_shared, _set_weight, find_snapshot,
)
sys.path.insert(0, "/home/gula/Documentos/Proyectos Github/prism-spectral-quantization/scripts")
from quantize_and_measure_ppl import MODEL_NAME, DEVICE, held_out_batches, MATRICES

N_LAYERS = 28


@torch.no_grad()
def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    seqs = held_out_batches(tok)
    print(f"[PRISM] held-out: {len(seqs)} secuencias, largos {[s.shape[1] for s in seqs]}", flush=True)

    snapshot_dir = find_snapshot(MODEL_NAME)
    reader = ShardReader(snapshot_dir)
    model, cfg = build_streaming_model(MODEL_NAME)
    materialize_shared(model, reader, embed_device=DEVICE, head_device=DEVICE, norm_device=DEVICE)
    _set_weight(model.model.norm, "weight", model.model.norm.weight.float())
    layers = model.model.layers
    rotary = model.model.rotary_emb
    head_dtype = model.lm_head.weight.dtype

    H = [model.model.embed_tokens(s.to(DEVICE)).float() for s in seqs]

    def causal_mask(S, dtype):
        m = torch.full((S, S), float("-inf"), dtype=dtype)
        return torch.triu(m, diagonal=1).view(1, 1, S, S)

    masks = [causal_mask(s.shape[1], torch.float32) for s in seqs]
    pos_embs = [rotary(H[j], torch.arange(seqs[j].shape[1]).unsqueeze(0)) for j in range(len(seqs))]

    meta = torch.device("meta")
    for layer_idx in range(N_LAYERS):
        blk = layers[layer_idx]
        pre = f"model.layers.{layer_idx}."
        _set_weight(blk.input_layernorm, "weight", reader.get(f"{pre}input_layernorm.weight").float().to(DEVICE))
        _set_weight(blk.post_attention_layernorm, "weight",
                    reader.get(f"{pre}post_attention_layernorm.weight").float().to(DEVICE))
        for hf_sub, dir_name in MATRICES:
            sub, proj = hf_sub.split(".")
            lin = getattr(getattr(blk, sub), proj)
            W = reader.get(f"{pre}{hf_sub}.weight").to(DEVICE).float()
            _set_weight(lin, "weight", W)
            bname = f"{pre}{hf_sub}.bias"
            if reader.has(bname):
                _set_weight(lin, "bias", reader.get(bname).to(DEVICE).float())

        for j in range(len(seqs)):
            out = blk(H[j], attention_mask=masks[j], position_embeddings=pos_embs[j])
            H[j] = out[0] if isinstance(out, tuple) else out

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
        print(f"[PRISM] base: capa {layer_idx+1}/{N_LAYERS} lista", flush=True)

    total_loss, total_tok = 0.0, 0
    for j, s in enumerate(seqs):
        h = model.model.norm(H[j])
        logits = model.lm_head(h.to(head_dtype)).float()
        labels = s.to(DEVICE)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="sum")
        total_loss += loss.item()
        total_tok += shift_labels.numel()
    ppl = float(torch.exp(torch.tensor(total_loss / total_tok)))
    print(f"[PRISM] base_ppl={ppl:.4f} tokens={total_tok}", flush=True)


if __name__ == "__main__":
    main()
