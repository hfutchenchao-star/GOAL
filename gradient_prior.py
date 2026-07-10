"""Compute the refusal-direction subspace from harmful examples.

For every harmful sample we run a forward + backward, flatten the gradient
over the trainable (middle-layer) parameters, and stack them. The mean is
the gradient prior; the top-k left singular vectors of the (centered or raw)
gradient matrix span the refusal subspace.
"""
import os
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import cfg
from data import SFTDataset, collate, load_harmful_only
from model_utils import (
    assign_flat_to_grads,
    build_trainable_model,
    flatten_grads,
    load_tokenizer,
)


@torch.no_grad()
def _zero_grads(params):
    for p in params:
        if p.grad is not None:
            p.grad.zero_()


def compute_refusal_basis():
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = load_tokenizer()
    model, params = build_trainable_model()
    model.to(device)
    model.train()  # we need grads, but we don't step the optimizer

    examples = load_harmful_only()[: cfg.prior_max_samples]
    print(f"[prior] using {len(examples)} harmful samples")
    ds = SFTDataset(examples, tok, cfg.max_seq_len)
    loader = DataLoader(
        ds, batch_size=cfg.prior_batch_size, shuffle=False,
        collate_fn=lambda b: collate(b, tok.pad_token_id),
    )

    expected_D = sum(p.numel() for p in params)
    pbar = tqdm(loader, desc="prior", dynamic_ncols=True)

    if cfg.use_mean_only:
        # Streaming mean: avoids stacking N×D gradients in RAM.
        mean_grad = torch.zeros(expected_D, dtype=torch.float32)
        n = 0
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            _zero_grads(params)
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            out.loss.backward()
            g = flatten_grads(params).cpu().float()
            # incremental mean: mean_{n+1} = mean_n * n/(n+1) + g/(n+1)
            mean_grad.mul_(n / (n + 1)).add_(g, alpha=1.0 / (n + 1))
            n += 1
            pbar.set_postfix(loss=f"{out.loss.item():.3f}")

        payload = {
            "mean_grad": mean_grad.half(),
            "n_samples": n,
            "dim":       expected_D,
            "mode":      "mean",
        }

        norm = mean_grad.norm()
        print(f"[prior] mean_grad norm = {norm.item():.6f}")
        if norm.item() == 0.0:
            raise RuntimeError("mean gradient has zero norm; cannot normalize.")
        basis = (mean_grad / norm).unsqueeze(0)   # (1, D), unit row
        payload["basis"] = basis
        payload["mean_norm"] = norm
    else:
        # SVD mode: need to stack all grads for SVD decomposition.
        grad_vectors = []
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            _zero_grads(params)
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            out.loss.backward()
            g = flatten_grads(params).cpu()
            grad_vectors.append(g)
            pbar.set_postfix(loss=f"{out.loss.item():.3f}")

        G = torch.stack(grad_vectors, dim=0)  # (N, D)
        mean_grad = G.mean(dim=0).float()     # (D,)

        payload = {
            "mean_grad": mean_grad.half(),
            "n_samples": G.shape[0],
            "dim":       G.shape[1],
            "mode":      "svd",
        }

        print("[prior] running SVD ...")
        U_lr, S, Vh = torch.linalg.svd(G.float(), full_matrices=False)
        k = min(cfg.refusal_subspace_k, Vh.shape[0])
        basis = Vh[:k].half()                     # (k, D), float16 to save disk
        print(f"[prior] singular values (top {k}): {S[:k].tolist()}")
        payload["basis"] = basis
        payload["sigma"] = S[:k].half()

    os.makedirs(os.path.dirname(cfg.prior_save_path), exist_ok=True)
    torch.save(payload, cfg.prior_save_path)
    print(f"[prior] saved -> {cfg.prior_save_path}  "
          f"(mode={payload['mode']}, basis={tuple(basis.shape)})")


if __name__ == "__main__":
    compute_refusal_basis()
