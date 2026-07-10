"""Dataset loading & label assignment.

Each example carries an integer `label` so the trainer knows whether to
orthogonalize its gradient against the refusal subspace.
"""
import json
import os
from typing import List, Dict

import torch
from torch.utils.data import Dataset

from config import cfg


def _load_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_labeled_examples() -> List[Dict]:
    """Return a single list mixing all three sources, each with a `label`."""
    out = []
    spec = [
        (cfg.harmful_file,      cfg.label_harmful),
        (cfg.harmless_file,     cfg.label_harmless),
        (cfg.over_refusal_file, cfg.label_over_refusal),
    ]
    for fname, label in spec:
        path = os.path.join(cfg.dataset_dir, fname)
        for ex in _load_json(path):
            instr = ex.get("instruction")
            output = ex.get("output")
            if not instr or not output:
                # Skip rows that haven't been filled by fill_outputs.py.
                continue
            out.append({"instruction": instr, "output": output, "label": label})
    return out


def load_harmful_only() -> List[Dict]:
    path = os.path.join(cfg.dataset_dir, cfg.harmful_file)
    rows = []
    for ex in _load_json(path):
        instr = ex.get("instruction")
        output = ex.get("output")
        if not instr or not output:
            continue
        rows.append({"instruction": instr, "output": output,
                     "label": cfg.label_harmful})
    return rows


class SFTDataset(Dataset):
    """Tokenizes (instruction, output) into causal-LM training tensors.

    Loss is masked over the prompt portion so we only learn the assistant
    response. The example's `label` is preserved on each item.
    """

    def __init__(self, examples: List[Dict], tokenizer, max_len: int):
        self.examples = examples
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        # Build prompt via the model's chat template (Llama-3.2-Instruct).
        prompt_msgs = [{"role": "user", "content": ex["instruction"]}]
        prompt_text = self.tok.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        full_text = prompt_text + ex["output"] + self.tok.eos_token

        prompt_ids = self.tok(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids   = self.tok(full_text,   add_special_tokens=False)["input_ids"]
        full_ids   = full_ids[: self.max_len]

        labels = list(full_ids)
        # Mask the prompt portion so loss is only on the response.
        mask_to = min(len(prompt_ids), len(labels))
        for i in range(mask_to):
            labels[i] = -100

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels":    torch.tensor(labels,   dtype=torch.long),
            "label_id":  torch.tensor(ex["label"], dtype=torch.long),
        }


def collate(batch, pad_id: int):
    """Right-pad a batch and stack the per-sample label_id."""
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids, labels, attn, label_ids = [], [], [], []
    for item in batch:
        ids = item["input_ids"]
        lab = item["labels"]
        pad = max_len - ids.size(0)
        input_ids.append(torch.cat([ids, torch.full((pad,), pad_id, dtype=torch.long)]))
        labels.append(torch.cat([lab, torch.full((pad,), -100, dtype=torch.long)]))
        attn.append(torch.cat([torch.ones_like(ids), torch.zeros(pad, dtype=torch.long)]))
        label_ids.append(item["label_id"])
    return {
        "input_ids":      torch.stack(input_ids),
        "labels":         torch.stack(labels),
        "attention_mask": torch.stack(attn),
        "label_ids":      torch.stack(label_ids),
    }
