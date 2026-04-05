# CIDA: Collective Intelligence via Deliberation and Aggregation

> *"What if a neural network could argue with itself — and reach a better answer?"*

**Author:** Kairat Zhaksylykov · K.Zhubanov Regional University · `zhaksylykov.k06@gmail.com`  
**Date:** April 2, 2026  
**Paper:** [`assets/CIDA_research_paper.pdf`](assets/CIDA_research_paper.pdf)

---

## Overview

Modern neural networks are often **miscalibrated**: they assign overconfident probabilities even when they are wrong. BERT-tiny, for example, produces an Expected Calibration Error (ECE) of 9.09% on SST-2 — meaning its confidence is systematically inflated.

**CIDA** (Collective Intelligence via Deliberation and Aggregation) addresses this at the architectural level by embedding a **multi-agent deliberation protocol** directly into the model. Instead of a single forward pass, *N* agents form independent perspectives, exchange compressed arguments (**theses**), refine their beliefs over *R* rounds, and reach a consensus weighted by each agent's own uncertainty.

No post-hoc calibration. No ensemble bloat. Just better-reasoned predictions — by design.

---

## Key Results

| Model | Params | Accuracy | ECE | Robustness |
|---|---|---|---|---|
| BERT-tiny (baseline) | 4.4M | 83.60% | 9.09% | 70.99% |
| **CIDA-BERT** | 5.3M | 83.14% | **5.63%** ↓38% | 68.58% |
| CIDA V8 Tiny (no pretrain) | 1.25M | 76.66% | **3.41%** ↓62% | 70.13% |

- CIDA-BERT reduces ECE by **38%** with only **0.46% accuracy loss**
- CIDA V8 Tiny (trained from scratch, 4× fewer params than BERT-tiny) achieves the **lowest ECE of all**
- A parameter-matched wide baseline yields **zero improvement** — the gain is architectural, not parametric

---

## Architecture

The full CIDA pipeline consists of six sequential modules:

```
Input Text
    │
    ▼
Token Embedding (BPE, vocab 16k)
    │
    ▼
Contextual Embedding (optional Transformer encoder)
    │
    ▼
Slot Attention Perspective Generator  ──→  N Initial Agent Beliefs (A_i^0)
    │
    ▼
Deliberation Layer × R rounds
   ├─ Sparse Token Communication  (K=4 theses per agent, cross-attention)
   ├─ AgentLoRA                   (rank-8 low-rank adaptation per agent)
   ├─ Sparse Mixture of Experts   (top-2 of 4 experts, SwiGLU)
   └─ Gated Update                (GRU-style state update)
    │
    ▼
Uncertainty Heads  →  var_i = softplus(Linear(b_i)) + ε
    │
    ▼
Consensus Aggregation  →  w_i ∝ (1/var_i) × learnable trust
    │
    ▼
MetaReviewer  →  Class Logits
```

### Full Architecture Diagram

![CIDA Full Architecture](assets/CIDA_full_architecture.jfif)

---

### Part 1 — From Text to Initial Agent Perspectives

The pipeline begins with **Byte-Pair Encoding** (vocab 16k), followed by an optional **Transformer Encoder** block (MHSA + FFN + LayerNorm). The **Slot Attention Perspective Generator** then uses *competitive* attention (softmax over slots, not positions) to assign each agent a distinct semantic slice of the input.

> Agents compete for the input. If one agent attends strongly to a token, others have less access — enforcing specialisation.

![Part 1: Encoder & Slot Attention](assets/CIDA_part_1_Encoder_Slot.jfif)

---

### Part 2 — The Deliberation Group (Agent Iterative Refinement)

Over *R* rounds, each agent:
1. **Compresses** its belief into K=4 theses (linear bottleneck)
2. **Attends** to all agents' theses via cross-attention (sparse, top-K mask)
3. **Adapts** via AgentLoRA (rank-8 matrices unique to each agent)
4. **Routes** through a Sparse MoE (top-2 of 4 SwiGLU experts)
5. **Updates** its state via a GRU-style gated cell

Weights are **tied across rounds** — deliberation becomes an iterative algorithm, not a chain of separate layers.

![Part 2: Deliberation Group](assets/CIDA_part_2_Deliberition_group.jfif)

---

### Part 3 — Consensus and Inference

After the final round, each agent projects its belief to a **feature vector** and estimates its own **predictive uncertainty** (via Softplus + ε). The Consensus Aggregation module:

$$w_i \propto \frac{1}{\text{var}_i} \times \text{trust}_i$$

computes an **inverse-variance weighted sum**, giving agents with lower uncertainty greater influence. A residual connection from the encoder's CLS token provides a skip path. The **MetaReviewer** outputs final class logits.

![Part 3: Consensus & Inference](assets/CIDA_part_3_Consensus_Inference.jfif)

---

## Theoretical Guarantees

| Claim | Result |
|---|---|
| **Optimal weighting** | Inverse-variance weights minimise consensus variance (min-variance linear unbiased estimator) |
| **Convergence** | GRU deliberation is a contraction mapping → converges exponentially to fixed point |
| **Variance reduction** | Multi-agent averaging reduces predictive variance by up to **1/N** |
| **Calibration** | Consensus distribution minimises expected KL divergence to the true distribution (convexity of KL) |

---

## Loss Function

$$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda_{\text{conv}}\mathcal{L}_{\text{conv}} + \lambda_{\text{nt}}\mathcal{L}_{\text{nt}} + \lambda_{\text{div}}\mathcal{L}_{\text{div}} + \lambda_{\text{dom}}\mathcal{L}_{\text{dom}} + \lambda_{\text{ponder}}\mathcal{L}_{\text{ponder}}$$

| Loss term | Weight | Purpose |
|---|---|---|
| $\mathcal{L}_{\text{task}}$ | 1.0 | Label-smoothed cross-entropy (ε=0.05) |
| $\mathcal{L}_{\text{conv}}$ | 0.5 | Penalise belief drift in later rounds (encourage convergence) |
| $\mathcal{L}_{\text{nt}}$ | 0.1 | Enforce non-trivial first-round change (margin 0.35) |
| $\mathcal{L}_{\text{div}}$ | 0.2 | Penalise pairwise cosine similarity (agent diversity) |
| $\mathcal{L}_{\text{dom}}$ | 0.1 | Penalise weight dominance (uniform consensus) |
| $\mathcal{L}_{\text{ponder}}$ | 0.01 | Encourage early termination (ACT-style) |

---

## Ablation Study

*All variants use the simple token embedder (no pre-training). Mean over seeds {42, 123}.*

| Variant | Params | Accuracy | ECE | Robustness |
|---|---|---|---|---|
| Baseline (no CDP) | 1.04M | 50.92% | 5.69% | 50.92% |
| **CDP full** | 1.25M | **76.66%** | **3.41%** | **70.13%** |
| No communication | 1.23M | 77.01% | 4.42% ↑30% | 68.52% |
| No diversity loss | 1.25M | 77.41% | 3.58% | 70.41% |
| No dominance loss | 1.25M | 76.49% | 3.68% | 70.58% |
| Agents = 2 | 1.23M | 77.41% | **3.40%** | **70.93%** |
| Agents = 8 | 1.26M | 77.75% | 3.96% | 70.76% |

**Key finding:** Removing Sparse Token Communication worsens ECE by **30%** — argument exchange is the single most important component for calibration.

---

## Training Details

```
Optimizer:    AdamW
LR:           2e-5 (CIDA-BERT)  |  3e-4 (CIDA V8 Tiny)
Batch size:   64 (BERT)         |  256 (Tiny)
Epochs:       30
Early stop:   patience = 8
Hardware:     Single NVIDIA RTX 3050 (laptop)
```

---

## Repository Structure

```
CIDA/
├── assets/
│   ├── CIDA_full_architecture.jfif
│   ├── CIDA_part_1_Encoder_Slot.jfif
│   ├── CIDA_part_2_Deliberition_group.jfif
│   ├── CIDA_part_3_Consensus_Inference.jfif
│   └── CIDA_research_paper.pdf
├── README.md
└── ...
```

---

## Citation

```bibtex
@article{zhaksylykov2026cida,
  title     = {CIDA: Collective Intelligence via Deliberation and Aggregation},
  author    = {Zhaksylykov, Kairat},
  year      = {2026},
  month     = {April},
  note      = {K.Zhubanov Regional University}
}
```

---

## Limitations

- Main experiments use **2 random seeds** (5 seeds + confidence intervals planned)
- CIDA-BERT vs. BERT-tiny comparison is **not parameter-matched** (MLP baseline of equal size needed)
- Results primarily on **SST-2**; generalisation to MNLI, QQP not yet verified
- CDP component contributions may differ when combined with larger pre-trained encoders

---

## Related Work

| Work | Relation to CIDA |
|---|---|
| Transformer [Vaswani et al., 2017] | Base architecture; heads operate independently — CIDA adds explicit inter-agent communication |
| Temperature Scaling [Guo et al., 2017] | Post-hoc calibration; does not change representations — CIDA is endogenous |
| Slot Attention [Locatello et al., 2020] | Adapted from vision to NLP; slots become competing semantic agents |
| Deep Ensembles [Lakshminarayanan et al., 2017] | Improve calibration but multiply inference cost — CIDA is a single model |
| Multi-Agent Debate [Du et al., 2024] | Post-hoc LLM debate — CIDA trains the full debate end-to-end in latent space |
| Switch Transformers / Mixtral | Sparse MoE for capacity — CIDA uses local MoE per agent per round |

---

*All experiments are fully reproducible on a single consumer GPU.*