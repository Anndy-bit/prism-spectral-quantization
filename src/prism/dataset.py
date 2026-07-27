"""Held-out evaluation batches.

Uses the same slice of the Alpaca training split, with the same prompt
template, as the rest of the lowrank-field-adapters project, so perplexity
numbers are comparable across both repositories.
"""

import torch

from prism.config import N_HELDOUT, MAX_SEQ


def held_out_batches(tokenizer, n_heldout: int = N_HELDOUT, max_seq: int = MAX_SEQ):
    from datasets import load_dataset

    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.select(range(len(ds) - n_heldout, len(ds)))

    batches = []
    for example in ds:
        text = (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Response:\n{example.get('output', '')}"
        )
        ids = tokenizer.encode(text, truncation=True, max_length=max_seq)
        if len(ids) > 2:
            batches.append(torch.tensor(ids).unsqueeze(0))
    return batches
