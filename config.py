"""Central config for the orthogonal-gradient fine-tuning project."""
import os
from dataclasses import dataclass, field
from typing import List


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Config:
    # ---- model ----
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    # Llama-2-7b-chat-hf has 32 transformer layers. Train layers [15, 32).
    middle_layer_start: int = 11
    middle_layer_end: int = 22 # exclusive
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    # Baseline switches:
    #   apply_lora_globally=True       -> LoRA on ALL layers (not just middle)
    #   use_orthogonal_projection=False -> skip prior loading & projection (plain SFT)
    # Set both for the global-LoRA SFT baseline experiment.
    apply_lora_globally: bool = False
    use_orthogonal_projection: bool = True

    # ---- data ----
    dataset_dir: str = os.environ.get(
        "GOAL_DATASET_DIR",
        os.path.join(PROJECT_DIR, "datasets"),
    )
    harmful_file: str = "harmful_train.json"
    harmless_file: str = "benign_train.json"
    over_refusal_file: str = "over-refusal_train.json"
    max_seq_len: int = 3048

    # Label convention.
    label_harmful: int = 0
    label_harmless: int = 1
    label_over_refusal: int = 2

    # ---- gradient prior ----
    prior_batch_size: int = 1          # per-sample grads, keep at 1
    prior_max_samples: int = 200      # cap on harmful samples used for prior
    # If True: basis = mean_grad / ||mean_grad||  (single direction, k=1).
    # If False: SVD on stacked per-sample grads, take top `refusal_subspace_k` rows.
    use_mean_only: bool = False
    refusal_subspace_k: int = 10     # only used when use_mean_only=False
    # Dimension of safe subspace built via SVD on G_harmless.
    # -1 = full rank (= n_harmless). 0 = use streaming mean (legacy q=1 path).
    # Any positive int = SVD top-q of harmless.
    # Only consulted when use_mean_only=False and use_full_rank=False.
    safe_subspace_q: int = -1
    # If True: subtract FULL safe subspace (rank=N harmless) from G_harm,
    # take FULL rank residual as refusal basis. Overrides use_mean_only.
    use_full_rank: bool = False
    prior_save_path: str = os.environ.get(
        "GOAL_PRIOR_PATH",
        os.path.join(PROJECT_DIR, "checkpoints", "refusal_basis-svd.pt"),
    )

    # ---- circle contrastive loss ----
    circle_loss_weight: float = 0.1   # λ for L_total = L_sft + λ * L_circle
    circle_push_only: bool = False     # True: only push apart negatives; False: also pull positives
    circle_margin: float = 0.25
    circle_gamma: float = 80
    circle_bank_size: int = 128        # class-balanced representation bank size
    # -1 means the final hidden-state layer returned by the model.
    circle_target_layer: int = -1
    # Rebuttal control: also remove the refusal-subspace component from the
    # contrastive gradient, i.e. use (I-P) g_cont instead of g_cont.
    project_contrastive_gradient: bool = os.environ.get(
        "GOAL_PROJECT_CONTRASTIVE_GRADIENT", "0"
    ).lower() in {"1", "true", "yes"}

    # ---- training ----
    output_dir: str = ""  # auto-generated in __post_init__
    epochs: int = 5
    micro_batch_size: int = 1         # samples backwarded individually then summed
    gradient_accumulation_steps: int = 4  # effective batch = micro_batch_size * this
    lr: float = 1.75e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    grad_clip: float = 1.0
    log_every: int = 10
    save_every: int = 1
    seed: int = 42
    bf16: bool = True

    # ---- logging ----
    use_wandb: bool = False
    wandb_project: str = "SVD Ortho + Circle Loss"
    wandb_run_name: str = ""   # leave blank to auto-name

    def __post_init__(self):
        if not self.output_dir:
            bs = f"bs{self.micro_batch_size}x{self.gradient_accumulation_steps}"
            if (not self.use_orthogonal_projection) and self.apply_lora_globally:
                # Baseline experiment: vanilla global-LoRA SFT.
                circle = f"circle{self.circle_loss_weight}"
                self.output_dir = os.path.join(
                    PROJECT_DIR,
                    "checkpoints",
                    f"baseline_sft_global-lora_{circle}_{bs}",
                )
            else:
                proj = "mean" if self.use_mean_only else f"svd-k{self.refusal_subspace_k}"
                circle = f"circle{self.circle_loss_weight}"
                if self.circle_loss_weight > 0:
                    circle += "-pushonly" if self.circle_push_only else "-full"
                    layer_tag = "final" if self.circle_target_layer == -1 else self.circle_target_layer
                    circle += f"-tl{layer_tag}-g{self.circle_gamma}"
                    if self.project_contrastive_gradient:
                        circle += "-projected-gradient"
                layers = f"L{self.middle_layer_start}-{self.middle_layer_end}"
                tune = "lora" if self.use_lora else "full"
                if self.apply_lora_globally:
                    tune = "lora-global"
                if not self.use_orthogonal_projection:
                    proj = "noproj"
                self.output_dir = os.path.join(
                    PROJECT_DIR,
                    "checkpoints",
                    f"{proj}_{circle}_{layers}_{tune}_{bs}",
                )


# ---- Pre-defined model configs ----
MODEL_CONFIGS = {
    "llama": {
        "model_name": "meta-llama/Llama-2-7b-chat-hf",
        "middle_layer_start": 11,
        "middle_layer_end": 22,
    },
    "qwen": {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "middle_layer_start": 13,
        "middle_layer_end": 22,
    },
        "llama3": {
        "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "middle_layer_start": 11,
        "middle_layer_end": 22,
    },                                                                                                                                                     
      "gemma": {
          "model_name": "google/gemma-7b-it",
          "middle_layer_start": 14,   
          "middle_layer_end": 23,     
    },
    "llama2_13b": {
        "model_name": "meta-llama/Llama-2-13b-chat-hf",
        "middle_layer_start": 12,   
        "middle_layer_end": 26,     
    },
    "llama3_1b": {
        "model_name": "meta-llama/Llama-3.2-1B-Instruct",
        "middle_layer_start": 7,    
        "middle_layer_end": 13,     
    },
}

cfg = Config()
