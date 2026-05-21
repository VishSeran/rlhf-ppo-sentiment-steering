# 🎭 rlhf-ppo-sentiment — Reinforcement Learning from Human Feedback Using PPO

> Fine-tune GPT-2 with Proximal Policy Optimization (PPO) and sentiment-based rewards to steer language model outputs toward **positive** or **negative** movie reviews — demonstrating a complete RLHF pipeline on the IMDb dataset.

---

## 📌 Repository Title

**`rlhf-ppo-sentiment`**

## 📝 Repository Description

A hands-on implementation of **Reinforcement Learning from Human Feedback (RLHF)** using **Proximal Policy Optimization (PPO)** to fine-tune the `lvwerra/gpt2-imdb` language model. The reward signal is derived from a DistilBERT sentiment classifier trained on IMDb movie reviews. Two models are trained — a **"Happy LLM"** (optimized for positive sentiment) and a **"Pessimistic LLM"** (optimized for negative sentiment) — and compared side-by-side to illustrate how RL-based alignment shapes language model behavior.

---

## 📚 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [How It Works](#how-it-works)
  - [1. Model & Tokenizer Initialization](#1-model--tokenizer-initialization)
  - [2. Dataset Preparation](#2-dataset-preparation)
  - [3. Reward Function](#3-reward-function)
  - [4. PPO Training Loop](#4-ppo-training-loop)
  - [5. Model Comparison](#5-model-comparison)
- [Results](#results)
- [Pretrained Models](#pretrained-models)
- [Dependencies](#dependencies)
- [References](#references)
- [Authors](#authors)
- [License](#license)

---

## Overview

This project demonstrates a complete **RLHF pipeline** where:

1. A base language model (`lvwerra/gpt2-imdb`) generates movie review continuations.
2. A sentiment analysis pipeline (`lvwerra/distilbert-imdb`) scores the generated text.
3. PPO uses those scores as rewards to update the language model's policy.
4. The result is two fine-tuned models with controllably different output tendencies — one biased toward positive language, one toward negative.

This mirrors the real-world alignment techniques used to train helpful, harmless, and honest AI assistants — just simplified to a sentiment-steering demonstration.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      RLHF PPO Loop                      │
│                                                         │
│  ┌──────────────┐     generate()     ┌───────────────┐  │
│  │  Query Text  │ ─────────────────► │  GPT-2 (IMDB) │  │
│  │  (IMDb tok.) │                    │  Policy Model │  │
│  └──────────────┘                    └──────┬────────┘  │
│                                             │ response  │
│                                             ▼           │
│                                   ┌─────────────────┐   │
│                                   │ DistilBERT      │   │
│                                   │ Sentiment Pipe  │   │
│                                   │ (reward signal) │   │
│                                   └────────┬────────┘   │
│                                            │ reward     │
│                                            ▼            │
│                              ┌─────────────────────┐    │
│                              │   PPO Trainer Step  │    │
│                              │  (policy + value    │    │
│                              │   function update)  │    │
│                              └─────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

The **reference model** (a frozen copy of the base model) is kept alongside the policy model. KL divergence between the policy and reference model acts as a regularization term, preventing the model from drifting too far from its original behavior.

---

## Project Structure

```
rlhf-ppo-sentiment/
│
├── PPOTrainer-v1.ipynb        # Main notebook — full RLHF pipeline
│
├── models/
│   ├── ppo-good/              # "Happy LLM" — trained on POSITIVE sentiment reward
│   └── ppo-bad/               # "Pessimistic LLM" — trained on NEGATIVE sentiment reward
│
├── stats/
│   ├── ppo-good.pkl           # Training statistics for the positive model
│   └── ppo-bad.pkl            # Training statistics for the negative model
│
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## Setup & Installation

### Prerequisites

- Python 3.8+
- A CUDA-capable GPU is **strongly recommended** for the training loop (CPU training is supported but very slow)

### Install Dependencies

```bash
pip install datasets==3.2.0 trl==0.11 transformers==4.43.4 \
            nltk==3.9.1 rouge_score==0.1.2 matplotlib==3.10.0 numpy==1.26.0

pip install torch==2.8.0+cpu torchtext==0.18.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

pip install rich==13.7.1
```

> For GPU environments, replace the `+cpu` variants with the appropriate CUDA wheel from [pytorch.org](https://pytorch.org).

---

## How It Works

### 1. Model & Tokenizer Initialization

The base model is `lvwerra/gpt2-imdb` — a GPT-2 variant already fine-tuned on IMDb movie reviews. It is loaded twice: once as the **trainable policy model** (wrapped with a value head for PPO) and once as the **frozen reference model**.

```python
config = PPOConfig(model_name="lvwerra/gpt2-imdb", learning_rate=1.41e-5)

model     = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token
```

### 2. Dataset Preparation

The IMDb training split is loaded, filtered to reviews longer than 200 characters, and tokenized. A `LengthSampler` introduces variability in input token lengths (2–8 tokens), making the model more robust to diverse inputs.

```python
dataset = build_dataset(config)
# Each sample has: input_ids (tensor), query (decoded text), review (full text)
```

A custom `collator` function groups samples into batches compatible with `PPOTrainer`.

### 3. Reward Function

A **DistilBERT sentiment pipeline** (`lvwerra/distilbert-imdb`) scores each generated text. The POSITIVE class score is used as the reward:

- **Happy LLM**: rewards high POSITIVE scores → model learns to generate upbeat continuations.
- **Pessimistic LLM**: rewards high NEGATIVE scores → model learns to generate gloomy continuations.

```python
sentiment_pipe = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb")

# Extract reward signal
rewards = [torch.tensor(item["score"])
           for output in pipe_outputs
           for item in output
           if item["label"] == sentiment]  # "POSITIVE" or "NEGATIVE"
```

### 4. PPO Training Loop

Each epoch:

1. Sample a batch of queries from the dataset.
2. Generate responses from the policy model.
3. Score each `query + response` pair with the sentiment pipeline.
4. Call `ppo_trainer.step(query_tensors, response_tensors, rewards)` to update the policy.
5. Log statistics including total loss, KL divergence, mean reward, and value function metrics.

The KL divergence between the policy and reference model prevents catastrophic forgetting and reward hacking.

### 5. Model Comparison

The `compare_models_on_dataset` function runs both a trained model and a reference model on the same set of query prompts, decodes their responses, and computes sentiment scores for each — returning a side-by-side DataFrame for easy inspection:

| query | response (before) | response (after) | rewards (before) | rewards (after) |
|-------|-------------------|------------------|------------------|-----------------|
| ...   | ...               | ...              | 0.43             | 0.91            |

---

## Results

Training curves show:

- **Loss**: PPO total loss (policy loss + value loss) stabilizes over epochs.
- **Mean Reward**: The mean sentiment score climbs steadily as the model learns to generate text aligned with the target sentiment.

The "Happy LLM" (`model_1`) produces measurably more positive continuations than the reference model, while the "Pessimistic LLM" (`model_0`) skews toward negative language — confirming that PPO-based RLHF reliably steers generative model behavior.

---

## Pretrained Models

To skip GPU-intensive training, pretrained weights and statistics are provided:

| Model | Download |
|-------|---------|
| Happy LLM weights | `ppo-good-tar.gz` (IBM Cloud Object Storage) |
| Happy LLM stats | `ppo-good.pkl` |
| Pessimistic LLM weights | `ppo-bad-tar.gz` |
| Pessimistic LLM stats | `ppo-bad.pkl` |

The notebook includes `wget` commands and extraction code to load these automatically.

---

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| `transformers` | 4.43.4 | Model loading, tokenization, generation |
| `trl` | 0.11 | PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead |
| `datasets` | 3.2.0 | IMDb dataset loading and preprocessing |
| `torch` | 2.8.0 | Tensor operations, model training |
| `matplotlib` | 3.10.0 | Training curve visualization |
| `pandas` | — | Results tabulation |
| `tqdm` | — | Progress bars |
| `nltk` | 3.9.1 | Text utilities |
| `numpy` | 1.26.0 | Numerical operations |

---

## References

- [TRL Library — Hugging Face](https://github.com/lvwerra/trl)
- [Original GPT-2 Sentiment Notebook](https://github.com/huggingface/trl/blob/main/examples/notebooks/gpt2-sentiment.ipynb)
- Schulman et al., *Proximal Policy Optimization Algorithms* (2017) — [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- Christiano et al., *Deep Reinforcement Learning from Human Preferences* (2017) — [arXiv:1706.03741](https://arxiv.org/abs/1706.03741)
- [Text Classification with TorchText](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html)
- [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/pdf/1902.00751.pdf)

---

## Authors

- **Joseph Santarcangelo** — Ph.D. in Electrical Engineering, IBM. Research in ML, signal processing, and computer vision.
- **Ashutosh Sagar** — MS in CS candidate, Dalhousie University. Experience in NLP and Data Science.

**Contributors:** Hailey Quach (IBM Data Scientist, Concordia University)

---

## License

© Copyright IBM Corporation. All rights reserved.

This project is based on IBM Skills Network course material and is intended for educational use.
