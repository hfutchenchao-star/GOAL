# GCG and TAO-Attack evaluation

This directory contains the code used to evaluate Base and GOAL on
Qwen3.6-27B against GCG and TAO-Attack. The rebuttal experiment uses 50
harmful behaviors and 250 optimization steps for each attack. Attack success
rates are evaluated with the official HarmBench classifier.

## Input data

The input is a CSV file containing a `Goal` column and, optionally, a `Target`
column. If `Target` is absent, the scripts use a generic affirmative target
prefix. `Behavior` and `Category` columns are also preserved when present.

Set the shared model and data paths once:

```bash
export GOAL_QWEN36_MODEL=/path/to/Qwen3.6-27B
export GOAL_JAILBREAK_DATA=/path/to/harmful-behaviors.csv
mkdir -p attack_results
```

Use `--ckpt base` for the unmodified base model and pass the GOAL LoRA adapter
directory for the fine-tuned model.

## GCG

```bash
python experiments/jailbreak/gcg_qwen36.py \
  --ckpt base \
  --limit 50 \
  --steps 250 \
  --out attack_results/gcg_base.json

python experiments/jailbreak/gcg_qwen36.py \
  --ckpt /path/to/goal_adapter \
  --limit 50 \
  --steps 250 \
  --out attack_results/gcg_goal.json
```

## TAO-Attack

```bash
python experiments/jailbreak/tao_qwen36.py \
  --ckpt base \
  --limit 50 \
  --steps 250 \
  --out attack_results/tao_base.json

python experiments/jailbreak/tao_qwen36.py \
  --ckpt /path/to/goal_adapter \
  --limit 50 \
  --steps 250 \
  --out attack_results/tao_goal.json
```

## HarmBench evaluation

```bash
python experiments/jailbreak/judge_harmbench.py \
  attack_results/gcg_base.json \
  attack_results/gcg_goal.json \
  attack_results/tao_base.json \
  attack_results/tao_goal.json \
  --output-dir attack_results/harmbench_judged
```

The reported attack success rate is 0% for both Base and GOAL under both GCG
and TAO-Attack.
