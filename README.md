# Revisiting Over-refusal in Safety-oriented Large Language Models: A Latent Knowledge Perspective

## Overview

This repository provides the implementation of **GOAL**, which mitigates over-refusal in safety-aligned language models with refusal-orthogonal gradient projection and safety-aware contrastive learning.

## Setup

This code has been tested with Python 3.8 or above.

We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and setting up an environment:

```bash
conda create -y --name goal python=3.8
conda activate goal
```

Then install the required packages:

```bash
pip install -r requirements.txt
```

## Datasets

Please place the training and evaluation files under `datasets/`, or set `GOAL_DATASET_DIR` to your local data directory.

Expected training files:

```text
datasets
  ├── harmful_train.json
  ├── benign_train.json
  └── over-refusal_train.json
```

Each file should be a JSON list with the following fields:

```json
{
  "instruction": "User prompt",
  "output": "Target assistant response"
}
```

You can use a custom data directory with:

```bash
export GOAL_DATASET_DIR=/path/to/datasets
```

## Usage

First compute the refusal-gradient basis:

```bash
python gradient_prior.py
```

Then train GOAL:

```bash
python train.py
```

### Projecting the contrastive gradient

```bash
GOAL_PROJECT_CONTRASTIVE_GRADIENT=1 python train.py
```

The default value is `0`, which preserves the original GOAL update. When the
option is enabled, checkpoints are written to a separate directory whose name
contains `projected-gradient`. Training also logs
`contrastive_subspace_energy_ratio`.

### Recomputing the refusal subspace every epoch

For the periodic-recomputation experiment, run:

```bash
GOAL_RECOMPUTE_SUBSPACE_EACH_EPOCH=1 python train.py
```

At the beginning of every epoch, the trainer extracts a new refusal subspace
from the current model and uses it for that epoch's gradient projection. It
also reports the overlap between consecutive epoch subspaces. The default is
`0`, which keeps the original fixed-subspace training behavior. This option
can be combined with `GOAL_PROJECT_CONTRASTIVE_GRADIENT=1`.

By default, checkpoints and the refusal basis are saved under `checkpoints/`. To use a custom refusal-basis path:

```bash
export GOAL_PRIOR_PATH=/path/to/refusal_basis-svd.pt
```

For evaluation:

```bash
export OPENAI_API_KEY=your_api_key
python eval.py --ckpt /path/to/checkpoint
```

## Optimization-based jailbreak evaluation

The GCG and TAO-Attack scripts used for the rebuttal, together with the
HarmBench evaluation script, are provided in
[`experiments/jailbreak`](experiments/jailbreak).
