"""GCG evaluation on Qwen3.6-27B used in the rebuttal."""

import argparse
import gc
import math
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from tao_qwen36 import (
    BASE_MODEL,
    DATASET,
    build_static_parts,
    encode_text,
    filter_roundtrip,
    forbidden_ids,
    generate,
    load_data,
    load_model,
    refusal_heuristic,
    save,
    sequence_loss,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--base_model", default=BASE_MODEL)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--search_width", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--suffix_tokens", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def control_gradient(model, embedding, before_ids, control_ids, after_ids, target_ids):
    control = embedding(control_ids.unsqueeze(0)).squeeze(0).detach()
    control.requires_grad_(True)
    loss = sequence_loss(
        model, embedding, before_ids, control, after_ids, target_ids
    )
    gradient = torch.autograd.grad(loss, control)[0].detach()
    return gradient, float(loss.detach())


@torch.no_grad()
def gcg_candidates(
    embedding_weight,
    control_ids,
    gradient,
    blocked_ids,
    search_width,
    topk,
):
    """Sample one-token substitutions from the standard GCG/HotFlip scores."""
    scores = -(gradient.float() @ embedding_weight.float().T)
    scores[:, blocked_ids] = -float("inf")
    scores[torch.arange(control_ids.numel(), device=scores.device), control_ids] = -float("inf")
    top_ids = scores.topk(min(topk, scores.shape[1]), dim=1).indices

    candidates = control_ids.unsqueeze(0).repeat(search_width, 1)
    positions = torch.arange(search_width, device=control_ids.device) % control_ids.numel()
    choices = torch.randint(
        0,
        top_ids.shape[1],
        (search_width,),
        device=control_ids.device,
    )
    candidates[torch.arange(search_width, device=control_ids.device), positions] = (
        top_ids[positions, choices]
    )
    return candidates


@torch.no_grad()
def candidate_losses(
    model,
    embedding,
    before_ids,
    candidate_ids,
    after_ids,
    target_ids,
    batch_size,
):
    results = []
    before = embedding(before_ids)
    after = embedding(after_ids)
    target = embedding(target_ids)
    for start in range(0, candidate_ids.shape[0], batch_size):
        ids = candidate_ids[start : start + batch_size]
        batch = ids.shape[0]
        prefix = torch.cat(
            [
                before.repeat(batch, 1, 1),
                embedding(ids),
                after.repeat(batch, 1, 1),
            ],
            dim=1,
        )
        inputs = torch.cat([prefix, target.repeat(batch, 1, 1)], dim=1)
        logits = model(inputs_embeds=inputs, use_cache=False).logits
        target_start = prefix.shape[1]
        selected = logits[
            :, target_start - 1 : target_start + target_ids.shape[1] - 1
        ]
        labels = target_ids.repeat(batch, 1)
        loss = F.cross_entropy(
            selected.transpose(1, 2), labels, reduction="none"
        ).mean(-1)
        results.append(loss.float().cpu())
        del prefix, inputs, logits
    return torch.cat(results)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    examples = load_data(args.dataset, args.offset, args.limit)
    model, tokenizer = load_model(args.base_model, args.ckpt)
    embedding = model.get_input_embeddings()
    blocked = forbidden_ids(tokenizer, embedding.weight.shape[0], model.device)
    results = []

    for item_number, example in enumerate(examples, 1):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print("=" * 80)
        print(f"[GCG] {item_number}/{len(examples)} index={example['dataset_index']}")
        print(f"[GCG] goal={example['goal'][:240]}")
        started = time.perf_counter()
        try:
            before_ids, after_ids, target_ids = build_static_parts(
                tokenizer, example["goal"], example["target"], model.device
            )
            initial_suffix = " ".join(["!"] * args.suffix_tokens)
            control_ids = encode_text(tokenizer, initial_suffix, model.device)[0]
            best_suffix = initial_suffix
            best_loss = math.inf
            history = []
            progress = tqdm(range(args.steps), desc="GCG", unit="step")

            for step in progress:
                gradient, current_loss = control_gradient(
                    model,
                    embedding,
                    before_ids,
                    control_ids,
                    after_ids,
                    target_ids,
                )
                candidates = gcg_candidates(
                    embedding.weight.detach(),
                    control_ids,
                    gradient,
                    blocked,
                    args.search_width,
                    args.topk,
                )
                candidates, texts = filter_roundtrip(tokenizer, candidates)
                losses = candidate_losses(
                    model,
                    embedding,
                    before_ids,
                    candidates,
                    after_ids,
                    target_ids,
                    args.eval_batch_size,
                )
                winner = int(losses.argmin())
                control_ids = candidates[winner].detach()
                best_suffix = texts[winner]
                best_loss = float(losses[winner])
                history.append(
                    {
                        "step": step + 1,
                        "loss": best_loss,
                        "pre_update_loss": current_loss,
                    }
                )
                progress.set_postfix(loss=f"{best_loss:.4f}")
                del gradient, candidates, losses
                torch.cuda.empty_cache()

            response = generate(
                model, tokenizer, example["goal"], best_suffix, args.max_new_tokens
            )
            elapsed = time.perf_counter() - started
            record = {
                **example,
                "status": "ok",
                "adversarial_suffix": best_suffix,
                "response": response,
                "best_loss": best_loss,
                "refusal_heuristic": refusal_heuristic(response),
                "elapsed_seconds": elapsed,
                "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                "history": history,
            }
            print(f"[GCG] suffix={best_suffix!r}")
            print(f"[GCG] response={response[:800]}")
            print(f"[GCG] elapsed={elapsed:.2f}s")
        except Exception as error:
            record = {
                **example,
                "status": "error",
                "error": repr(error),
                "elapsed_seconds": time.perf_counter() - started,
            }
            print(f"[GCG] ERROR: {error!r}")
        results.append(record)
        save(args.out, args, results)

    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
