# 3. Fine-Tuning Methods Overview

There are three practical ways to actually update an LLM. This doc compares them
so you can defend your choice. Deep dives follow in docs 4 and 5.

## The big picture

```
              memory needed        what gets trained
Full FT        very high           ALL weights
LoRA           medium              tiny added "adapter" matrices
QLoRA          low                 tiny adapters, base frozen in 4-bit
```

## 1. Full fine-tuning

Update **every** parameter in the model.

- ✅ Maximum capacity to change behavior; best possible quality if done right.
- ❌ Needs enormous GPU memory (80GB+ for 7B), one full copy of weights per
  task, and risks **catastrophic forgetting** (the model loses general skills).
- **Use when:** you have lots of compute and data, and the task is a big shift.

## 2. PEFT — Parameter-Efficient Fine-Tuning (the family)

**PEFT** is an umbrella term for methods that freeze the base model and train
only a small number of *new* parameters. The Hugging Face library is literally
called `peft`. LoRA is the most popular PEFT method. Others: prefix tuning,
prompt tuning, IA³, adapters. **In interviews: "LoRA is a PEFT method."**

## 3. LoRA — Low-Rank Adaptation

Freeze the original weights. Inject small trainable **low-rank matrices** into
chosen layers and train only those. You end up with a tiny "adapter" file (a few
MB) instead of a full model copy.

- ✅ ~10,000× fewer trained parameters; adapters are tiny and swappable; little
  forgetting because base weights are untouched.
- ✅ Fits one 7B model + training on a single 24GB GPU.
- ❌ Slightly less expressive than full FT (rarely matters in practice).
- Details: [04_lora_explained.md](04_lora_explained.md)

## 4. QLoRA — Quantized LoRA

LoRA, but the frozen base model is loaded in **4-bit** to slash memory further,
while the small LoRA adapters train in 16-bit.

- ✅ Fine-tune a 7B model on a **free Colab T4 (16GB)**; ~33B on a single 24GB card.
- ✅ Quality very close to 16-bit LoRA thanks to NF4 + double quantization.
- ❌ A bit slower per step than plain LoRA (dequantization overhead).
- Details: [05_qlora_explained.md](05_qlora_explained.md)

## Decision guide

| Your situation | Use |
|----------------|-----|
| Free Colab / one consumer GPU, 7B+ model | **QLoRA** |
| One 24–48GB GPU, want max speed | **LoRA** |
| Multi-GPU cluster, big domain shift, lots of data | **Full FT** |
| Many tasks sharing one base model | **LoRA/QLoRA** (swap adapters) |

## What "rank" and "alpha" mean (preview)

- **r (rank):** size of the LoRA matrices. Higher r = more capacity + more
  params. Common: 8, 16, 32, 64.
- **lora_alpha:** a scaling factor for the adapter's effect. Common: 16 or 32.
- **target_modules:** which layers get adapters (usually the attention
  projections `q_proj`, `k_proj`, `v_proj`, `o_proj`).

You will set all of these in `src/config.py` and the notebooks.

➡️ Next: [04_lora_explained.md](04_lora_explained.md)
