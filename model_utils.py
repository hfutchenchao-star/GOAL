"""Model loading + middle-layer parameter selection.

We freeze everything except the middle third of transformer layers. If
`use_lora=True` we additionally wrap those layers' linear modules with LoRA
adapters and only train the LoRA parameters.
"""
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import cfg


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model():
    dtype = torch.bfloat16 if cfg.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    return model


def _middle_layer_indices() -> range:
    return range(cfg.middle_layer_start, cfg.middle_layer_end)


def _get_layer_modules(model) -> list:
    # Llama: model.model.layers is a ModuleList
    return model.model.layers


def freeze_to_middle_layers(model) -> None:
    """Set requires_grad=True only on middle-third layer parameters."""
    for p in model.parameters():
        p.requires_grad = False
    layers = _get_layer_modules(model)
    for i in _middle_layer_indices():
        for p in layers[i].parameters():
            p.requires_grad = True


def apply_lora_to_middle(model):
    """Wrap LoRA on linear modules. Default: middle layers only. When
    `cfg.apply_lora_globally=True`, drop the layer restriction so all
    transformer blocks get adapters (used for the SFT baseline)."""
    from peft import LoraConfig, get_peft_model

    lora_kwargs = dict(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg.lora_target_modules,
    )
    if not cfg.apply_lora_globally:
        lora_kwargs["layers_to_transform"] = list(_middle_layer_indices())
        lora_kwargs["layers_pattern"] = "layers"
    lora_cfg = LoraConfig(**lora_kwargs)
    model = get_peft_model(model, lora_cfg)
    return model


def build_trainable_model() -> Tuple[torch.nn.Module, List[torch.nn.Parameter]]:
    """Loads the model, freezes appropriately, returns (model, trainable_params)."""
    model = load_model()
    if cfg.use_lora:
        # PEFT freezes base params automatically; LoRA modules only on middle layers.
        model = apply_lora_to_middle(model)
    else:
        freeze_to_middle_layers(model)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n = sum(p.numel() for p in trainable)
    print(f"[model] trainable params: {n:,}")
    return model, trainable


def flatten_grads(params: List[torch.nn.Parameter]) -> torch.Tensor:
    """Flatten current `.grad` of `params` into a single 1D float32 vector."""
    chunks = []
    for p in params:
        if p.grad is None:
            chunks.append(torch.zeros(p.numel(), device=p.device, dtype=torch.float32))
        else:
            chunks.append(p.grad.detach().to(torch.float32).reshape(-1))
    return torch.cat(chunks)


def assign_flat_to_grads(flat: torch.Tensor, params: List[torch.nn.Parameter]) -> None:
    """Inverse of flatten_grads: write a flat vector back into per-param `.grad`."""
    offset = 0
    for p in params:
        n = p.numel()
        chunk = flat[offset: offset + n].reshape(p.shape)
        if p.grad is None:
            p.grad = chunk.to(p.dtype).clone()
        else:
            p.grad.copy_(chunk.to(p.dtype))
        offset += n
    assert offset == flat.numel()
