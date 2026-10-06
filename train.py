"""Fine-tune with SVD-based orthogonal projection + Circle contrastive loss.

Two-phase per-step design:
  Phase 1 — Batched forward to extract hidden-state representations; compute
            Circle Loss (over-refusal vs unsafe) against a class-balanced bank.
  Phase 2 — Per-sample backward for SFT loss with gradient projection:
            over-refusal gradients have their top-k SVD refusal-subspace
            component removed.

Default gradient = (1/B) Σ project(g_sft_i) + λ · g_circle
Control experiment = (1/B) Σ project(g_sft_i) + project(λ · g_circle)
"""
import math
import os
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import cfg
from data import SFTDataset, collate, load_harmful_only, load_labeled_examples
from model_utils import (
    assign_flat_to_grads,
    build_trainable_model,
    flatten_grads,
    load_tokenizer,
)

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def project_out(g: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Orthogonal projection: remove the full component of g in the refusal
    subspace.

        g_tilde = g - P g = g - Σ_j (u_j^T g) u_j

    g: (D,)   basis: (k, D)  rows orthonormal in parameter space.
    """
    coeffs = basis @ g
    return g - basis.t() @ coeffs


def subspace_energy_ratio(g: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Return ||P g||_2^2 / ||g||_2^2 for an orthonormal row basis."""
    denominator = g.square().sum()
    if denominator <= 0:
        return torch.zeros((), device=g.device, dtype=g.dtype)
    coeffs = basis @ g
    return coeffs.square().sum() / denominator


def subspace_overlap(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Mean squared cosine of the principal angles between two row spaces."""
    rank = min(first.shape[0], second.shape[0])
    return (first.float() @ second.float().t()).square().sum() / max(1, rank)


def recompute_refusal_basis(model, params, tokenizer, device, epoch: int):
    """Compute a refusal basis from the current model at an epoch boundary."""
    examples = load_harmful_only()[: cfg.prior_max_samples]
    if not examples:
        raise RuntimeError("No harmful examples available for subspace extraction")
    dataset = SFTDataset(examples, tokenizer, cfg.max_seq_len)
    loader = DataLoader(
        dataset,
        batch_size=cfg.prior_batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id),
    )
    expected_dim = sum(parameter.numel() for parameter in params)
    gradients = []
    mean_gradient = torch.zeros(expected_dim, dtype=torch.float32)
    count = 0
    progress = tqdm(
        loader,
        desc=f"subspace epoch {epoch + 1}",
        dynamic_ncols=True,
        leave=False,
    )

    for batch in progress:
        batch = {key: value.to(device) for key, value in batch.items()}
        for parameter in params:
            if parameter.grad is not None:
                parameter.grad.zero_()
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        output.loss.backward()
        gradient = flatten_grads(params).cpu().float()
        if cfg.use_mean_only:
            mean_gradient.mul_(count / (count + 1)).add_(
                gradient, alpha=1.0 / (count + 1)
            )
        else:
            gradients.append(gradient)
        count += 1
        progress.set_postfix(loss=f"{output.loss.item():.3f}")

    for parameter in params:
        if parameter.grad is not None:
            parameter.grad.zero_()

    if cfg.use_mean_only:
        norm = mean_gradient.norm()
        if norm.item() == 0.0:
            raise RuntimeError("Mean harmful gradient has zero norm")
        basis = (mean_gradient / norm).unsqueeze(0)
    else:
        gradient_matrix = torch.stack(gradients, dim=0)
        _, singular_values, right_vectors = torch.linalg.svd(
            gradient_matrix, full_matrices=False
        )
        rank = min(cfg.refusal_subspace_k, right_vectors.shape[0])
        basis = right_vectors[:rank]
        print(
            f"[train] epoch {epoch + 1} top-{rank} singular values: "
            f"{singular_values[:rank].tolist()}"
        )

    print(
        f"[train] recomputed refusal basis for epoch {epoch + 1}: "
        f"{tuple(basis.shape)} from {len(examples)} harmful examples"
    )
    return basis.half()


def last_token_pool(hidden_states: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
    """Select the last non-padding token hidden state for each sequence.

    hidden_states: (B, L, H)   attention_mask: (B, L)
    Returns: (B, H)
    """
    last_indices = attention_mask.long().sum(dim=1).clamp(min=1) - 1
    batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
    return hidden_states[batch_indices, last_indices]


# ---------------------------------------------------------------------------
# Circle Loss
# ---------------------------------------------------------------------------

class RepresentationBank:
    """Class-balanced FIFO bank of (representation, label) pairs."""

    def __init__(self, max_size: int, feat_dim: int, device: torch.device,
                 class_labels: list):
        self.num_classes = len(class_labels)
        self.per_class_size = max(1, max_size // self.num_classes)
        self.max_size = self.per_class_size * self.num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.class_labels = class_labels

        self.banks = {}
        self.ptrs = {}
        self.counts = {}
        for label in class_labels:
            self.banks[label] = torch.zeros(self.per_class_size, feat_dim,
                                            device=device)
            self.ptrs[label] = 0
            self.counts[label] = 0

    @torch.no_grad()
    def push(self, feats: torch.Tensor, labels: torch.Tensor):
        """Route each sample to its class-specific sub-queue."""
        for i in range(feats.size(0)):
            label = int(labels[i].item())
            if label not in self.banks:
                continue
            idx = self.ptrs[label]
            self.banks[label][idx] = feats[i]
            self.ptrs[label] = (idx + 1) % self.per_class_size
            self.counts[label] += 1

    def get(self):
        """Return all valid entries from all class sub-queues."""
        parts_feat = []
        parts_label = []
        for label in self.class_labels:
            n = min(self.counts[label], self.per_class_size)
            if n == 0:
                continue
            parts_feat.append(self.banks[label] if n == self.per_class_size
                              else self.banks[label][:n])
            parts_label.append(torch.full((n,), label, dtype=torch.long,
                                          device=self.device))
        if not parts_feat:
            return (torch.zeros(0, self.feat_dim, device=self.device),
                    torch.zeros(0, dtype=torch.long, device=self.device))
        return torch.cat(parts_feat), torch.cat(parts_label)


def compute_circle_loss(query_feats: torch.Tensor,
                        query_labels: torch.Tensor,
                        bank_feats: torch.Tensor,
                        bank_labels: torch.Tensor,
                        margin: float = 0.25,
                        gamma: float = 80):
    """Circle Loss between over-refusal and unsafe samples.

    query_feats has grad; bank_feats is detached.
    Returns scalar loss or None if not enough pairs.
    """
    label_or = cfg.label_over_refusal
    label_unsafe = cfg.label_harmful

    # Keep only the two contrastive classes
    q_mask = (query_labels == label_or) | (query_labels == label_unsafe)
    b_mask = (bank_labels == label_or) | (bank_labels == label_unsafe)

    if q_mask.sum() < 1 or b_mask.sum() < 2:
        return None

    q_feats = F.normalize(query_feats[q_mask], dim=1)   # (Q, H)
    q_labs  = query_labels[q_mask]                       # (Q,)
    b_feats = F.normalize(bank_feats[b_mask], dim=1)     # (K, H)
    b_labs  = bank_labels[b_mask]                        # (K,)

    sim  = q_feats @ b_feats.t()                         # (Q, K)
    same = q_labs.unsqueeze(1) == b_labs.unsqueeze(0)    # (Q, K)

    NEG_INF = -1e9

    if cfg.circle_push_only:
        # Only push apart negatives (异类推远)
        has_neg = (~same).any(dim=1)
        if not has_neg.any():
            return None

        on = -margin
        dn = margin
        alpha_n = torch.clamp(sim.detach() - on, min=0)
        logit_n = gamma * alpha_n * (sim - dn)
        logit_n = logit_n.masked_fill(same, NEG_INF)
        lse_n = torch.logsumexp(logit_n, dim=1)
        loss = F.softplus(lse_n)
        return loss[has_neg].mean()
    else:
        # Full circle loss: push apart negatives + pull together positives
        has_both = same.any(dim=1) & (~same).any(dim=1)
        if not has_both.any():
            return None

        op = 1 + margin
        on = -margin
        dp = 1 - margin
        dn = margin

        alpha_p = torch.clamp(op - sim.detach(), min=0)
        alpha_n = torch.clamp(sim.detach() - on, min=0)

        logit_p = -gamma * alpha_p * (sim - dp)
        logit_n =  gamma * alpha_n * (sim - dn)

        logit_p = logit_p.masked_fill(~same, NEG_INF)
        logit_n = logit_n.masked_fill(same, NEG_INF)

        lse_p = torch.logsumexp(logit_p, dim=1)
        lse_n = torch.logsumexp(logit_n, dim=1)

        loss = F.softplus(lse_p + lse_n)
        return loss[has_both].mean()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg.output_dir, exist_ok=True)

    if ((cfg.project_contrastive_gradient
         or cfg.recompute_refusal_subspace_each_epoch)
            and not cfg.use_orthogonal_projection):
        raise ValueError(
            "Contrastive-gradient projection and periodic subspace "
            "recomputation require use_orthogonal_projection=True"
        )

    # ---- model ----
    tok = load_tokenizer()
    model, params = build_trainable_model()
    model.to(device)
    model.train()

    # ---- refusal basis (SVD top-k); only needed if projection is on ----
    expected_D = sum(p.numel() for p in params)
    basis = None
    if cfg.use_orthogonal_projection:
        if cfg.recompute_refusal_subspace_each_epoch:
            print("[train] per-epoch refusal-subspace recomputation ENABLED")
        else:
            if not os.path.exists(cfg.prior_save_path):
                raise FileNotFoundError(
                    f"Refusal basis not found at {cfg.prior_save_path}. "
                    f"Run `python gradient_prior.py` first."
                )
            prior = torch.load(cfg.prior_save_path, map_location="cpu")
            basis = prior["basis"].to(device=device, dtype=torch.float32)
            assert basis.shape[1] == expected_D, (
                f"basis dim {basis.shape[1]} != trainable param dim {expected_D}. "
                f"Did the trainable parameter set change between prior and training?"
            )
            print(f"[train] loaded refusal basis: {tuple(basis.shape)} "
                  f"(mode={prior.get('mode', '?')})")
        if cfg.project_contrastive_gradient:
            print("[train] contrastive-gradient projection ENABLED")
    else:
        print("[train] orthogonal projection DISABLED (baseline SFT mode)")

    # ---- data ----
    examples = load_labeled_examples()
    random.shuffle(examples)
    print(f"[train] total mixed examples: {len(examples)}")
    ds = SFTDataset(examples, tok, cfg.max_seq_len)
    loader = DataLoader(
        ds, batch_size=cfg.micro_batch_size, shuffle=True,
        collate_fn=lambda b: collate(b, tok.pad_token_id),
    )

    # ---- optim ----
    micro_steps = len(loader) * cfg.epochs
    total_steps = micro_steps // cfg.gradient_accumulation_steps
    warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))
    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    def lr_at(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    # ---- representation bank for circle loss ----
    hidden_dim = model.config.hidden_size
    rep_bank = RepresentationBank(
        cfg.circle_bank_size,
        hidden_dim,
        device,
        class_labels=[cfg.label_harmful, cfg.label_over_refusal],
    )

    # ---- wandb ----
    use_wandb = cfg.use_wandb and _WANDB_AVAILABLE
    if cfg.use_wandb and not _WANDB_AVAILABLE:
        print("[train] wandb requested but not installed; `pip install wandb`.")
    if use_wandb:
        run_name = cfg.wandb_run_name or (
            f"svd-k{cfg.refusal_subspace_k}"
            f"-circle{cfg.circle_loss_weight}"
            f"-lr{cfg.lr}-ep{cfg.epochs}-bs{cfg.micro_batch_size}"
        )
        wandb.init(
            project=cfg.wandb_project,
            name=run_name,
            config={k: getattr(cfg, k) for k in vars(cfg)
                    if not k.startswith("_") and not callable(getattr(cfg, k))},
        )
        wandb.config.update({"total_steps": total_steps,
                             "warmup_steps": warmup_steps,
                             "n_train_examples": len(examples),
                             "trainable_params": expected_D})

    global_step = 0
    micro_step = 0
    grad_accum = torch.zeros(expected_D, device=device, dtype=torch.float32)
    accum_sft_loss = 0.0
    accum_cl_loss = 0.0
    accum_cl_subspace_ratio = 0.0
    accum_cl_subspace_count = 0
    accum_projected = 0
    accum_samples = 0
    previous_epoch_basis = None

    pbar = tqdm(total=total_steps, desc="train", dynamic_ncols=True)
    for epoch in range(cfg.epochs):
        if cfg.recompute_refusal_subspace_each_epoch:
            basis_cpu = recompute_refusal_basis(
                model, params, tok, device, epoch
            )
            assert basis_cpu.shape[1] == expected_D, (
                f"basis dim {basis_cpu.shape[1]} != trainable param dim {expected_D}"
            )
            overlap = None
            if previous_epoch_basis is not None:
                overlap = subspace_overlap(previous_epoch_basis, basis_cpu).item()
                print(
                    f"[train] epoch {epoch + 1} subspace overlap with previous "
                    f"epoch: {100 * overlap:.2f}%"
                )
            previous_epoch_basis = basis_cpu
            basis = basis_cpu.to(device=device, dtype=torch.float32)
            if use_wandb and overlap is not None:
                wandb.log(
                    {"train/subspace_overlap_previous_epoch": overlap},
                    step=global_step,
                )

        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            B = batch["input_ids"].size(0)

            # =============================================================
            # Phase 1: Circle contrastive loss (batched forward)
            # =============================================================
            contrastive_g = torch.zeros(expected_D, device=device,
                                        dtype=torch.float32)
            cl_loss_val = 0.0
            cl_subspace_ratio = None

            if cfg.circle_loss_weight > 0:
                for p in params:
                    if p.grad is not None:
                        p.grad.zero_()

                out_cl = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    output_hidden_states=True,
                )
                last_hidden = out_cl.hidden_states[cfg.circle_target_layer].float()
                reps = last_token_pool(last_hidden, batch["attention_mask"])  # (B, H)

                bank_feats, bank_labels = rep_bank.get()
                cl_loss = compute_circle_loss(
                    reps, batch["label_ids"],
                    bank_feats, bank_labels,
                    margin=cfg.circle_margin, gamma=cfg.circle_gamma,
                )

                if cl_loss is not None:
                    (cfg.circle_loss_weight * cl_loss).backward()
                    contrastive_g = flatten_grads(params).to(device)
                    cl_loss_val = cl_loss.item()
                    if basis is not None:
                        cl_subspace_ratio = subspace_energy_ratio(
                            contrastive_g, basis
                        ).item()
                    if cfg.project_contrastive_gradient:
                        contrastive_g = project_out(contrastive_g, basis)

                # Update bank with detached representations
                rep_bank.push(reps.detach(), batch["label_ids"])

            # =============================================================
            # Phase 2: SFT loss with per-sample gradient projection
            # =============================================================
            step_accum = torch.zeros(expected_D, device=device, dtype=torch.float32)
            total_loss = 0.0
            n_projected = 0

            for i in range(B):
                for p in params:
                    if p.grad is not None:
                        p.grad.zero_()

                out = model(
                    input_ids=batch["input_ids"][i:i + 1],
                    attention_mask=batch["attention_mask"][i:i + 1],
                    labels=batch["labels"][i:i + 1],
                )
                loss = out.loss
                if torch.isnan(loss):
                    continue
                loss.backward()
                total_loss += loss.item()

                g = flatten_grads(params).to(device)
                if (cfg.use_orthogonal_projection
                        and int(batch["label_ids"][i].item()) == cfg.label_over_refusal):
                    g = project_out(g, basis)
                    n_projected += 1
                step_accum.add_(g)

            step_accum.div_(B)

            # Combine projected SFT gradients with either the original or
            # refusal-orthogonal contrastive gradient.
            step_accum.add_(contrastive_g)

            # Accumulate into running gradient buffer
            grad_accum.add_(step_accum)
            accum_sft_loss += total_loss / max(1, B)
            accum_cl_loss += cl_loss_val
            if cl_subspace_ratio is not None:
                accum_cl_subspace_ratio += cl_subspace_ratio
                accum_cl_subspace_count += 1
            accum_projected += n_projected
            accum_samples += B
            micro_step += 1

            # ---- optimizer step every gradient_accumulation_steps ----
            if micro_step % cfg.gradient_accumulation_steps == 0:
                grad_accum.div_(cfg.gradient_accumulation_steps)

                assign_flat_to_grads(grad_accum, params)
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                for pg in optim.param_groups:
                    pg["lr"] = cfg.lr * lr_at(global_step)
                optim.step()
                optim.zero_grad(set_to_none=True)

                avg_loss = accum_sft_loss / cfg.gradient_accumulation_steps
                avg_cl = accum_cl_loss / cfg.gradient_accumulation_steps
                avg_cl_ratio = (
                    accum_cl_subspace_ratio / accum_cl_subspace_count
                    if accum_cl_subspace_count > 0 else 0.0
                )
                cur_lr = optim.param_groups[0]["lr"]

                pbar.update(1)
                pbar.set_postfix(epoch=epoch, sft=f"{avg_loss:.4f}",
                                 cl=f"{avg_cl:.4f}",
                                 cl_sub=f"{100 * avg_cl_ratio:.2f}%",
                                 lr=f"{cur_lr:.2e}",
                                 proj=f"{accum_projected}/{accum_samples}")

                if use_wandb:
                    wandb.log({
                        "train/sft_loss":    avg_loss,
                        "train/circle_loss": avg_cl,
                        "train/contrastive_subspace_energy_ratio": avg_cl_ratio,
                        "train/lr":          cur_lr,
                        "train/epoch":       epoch,
                        "train/n_projected_in_batch": accum_projected,
                        "train/batch_size":  accum_samples,
                    }, step=global_step)

                global_step += 1

                # Reset accumulators
                grad_accum.zero_()
                accum_sft_loss = 0.0
                accum_cl_loss = 0.0
                accum_cl_subspace_ratio = 0.0
                accum_cl_subspace_count = 0
                accum_projected = 0
                accum_samples = 0

                # End of epoch: save if requested.
        if cfg.save_every > 0 and (epoch + 1) % cfg.save_every == 0:
            _save(model, tok,
                  os.path.join(cfg.output_dir, f"epoch-{epoch + 1}"))

    pbar.close()
    _save(model, tok, os.path.join(cfg.output_dir, "final"))
    if use_wandb:
        wandb.finish()


def _save(model, tok, path: str):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tok.save_pretrained(path)
    print(f"[train] saved -> {path}")


if __name__ == "__main__":
    main()
