# 4. LoRA Explained From Scratch

LoRA (Low-Rank Adaptation) is the single most important technique in this
course. If you understand this doc, you can explain 80% of modern fine-tuning.

## The problem LoRA solves

Full fine-tuning updates a weight matrix `W` (millions of numbers) for every
layer. That means storing a full-size gradient and optimizer state for every
weight — huge memory, and a full model copy per task.

## The key insight

When you fine-tune, the *change* you make to the weights, call it `ΔW`, is
usually **low-rank** — it contains far less information than its size suggests.
A big matrix can be approximated by multiplying two skinny matrices.

So instead of learning the full `ΔW` (size `d × d`), LoRA learns two small
matrices `A` and `B`:

```
ΔW ≈ B · A
        where  A is (r × d)   and   B is (d × r),   with r << d
```

`r` is the **rank** — typically 8, 16, 32, or 64. If `d = 4096` and `r = 16`,
you go from `4096 × 4096 ≈ 16.7M` numbers to `2 × 4096 × 16 ≈ 131K` numbers per
matrix — about **128× fewer parameters**.

## How it works during training

```
       frozen                trainable
input ──► W  ──────────────►(+)──► output
      └──► A ──► B ──────────┘
              (scaled by alpha/r)
```

1. The original weight `W` is **frozen** (no gradients).
2. A parallel path computes `B · A · x`, scaled by `lora_alpha / r`.
3. The outputs are added: `h = Wx + (alpha/r)·BAx`.
4. Only `A` and `B` receive gradients → tiny memory footprint.
5. `A` is initialized random, `B` is initialized to **zero**, so at step 0 the
   adapter contributes nothing and the model behaves exactly like the base.

## The hyperparameters that matter

| Param | Meaning | Typical | Effect |
|-------|---------|---------|--------|
| `r` | rank of the adapter | 8–64 | higher = more capacity + more params |
| `lora_alpha` | scaling of the update | 16 or 32 | often set to `r` or `2r` |
| `lora_dropout` | dropout on adapter | 0.05–0.1 | regularization |
| `target_modules` | which layers get adapters | `q_proj,k_proj,v_proj,o_proj` (+ MLP) | more modules = more capacity |
| `bias` | train bias terms? | `"none"` | usually leave off |

**Rule of thumb:** start with `r=16, alpha=32`, targeting the attention
projections. Increase `r` if the model underfits.

## What you get out

A small **adapter** (often 10–200 MB) — just `A` and `B` for each targeted
layer, plus a config. You can:
- Ship the adapter alongside the base model and load both at inference, OR
- **Merge** the adapter back into `W` to get a standalone model
  (`src/merge_adapter.py`). Merging removes inference overhead but loses the
  swap-ability.

## Why LoRA barely forgets

Because the original weights are frozen, the model keeps all its general
abilities. You are *adding* a small skill, not overwriting the whole brain.

## Minimal code (you'll see this in the notebooks)

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()   # e.g. "trainable: 0.06% of all params"
```

➡️ Next: [05_qlora_explained.md](05_qlora_explained.md)
