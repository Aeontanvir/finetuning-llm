# 5. QLoRA & Quantization Explained

QLoRA = **Quantization + LoRA**. It is what lets you fine-tune a 7B model on a
free Colab GPU. This doc explains the quantization half (LoRA is in doc 4).

## What is quantization?

Quantization stores weights using fewer bits. A 7B model in FP16 is ~14 GB; in
4-bit it is ~3.5 GB. The trade-off is a small loss of precision, but with the
right scheme the quality cost is tiny.

## The three tricks that make QLoRA work

QLoRA (Dettmers et al., 2023) combines three ideas:

1. **4-bit NormalFloat (NF4).** A 4-bit datatype designed for weights that
   follow a normal distribution (which neural-net weights roughly do). It packs
   more useful precision into 4 bits than plain INT4.

2. **Double quantization.** The quantization itself needs small "scale"
   constants; QLoRA quantizes *those* too, saving a further ~0.4 bits/param.

3. **Paged optimizers.** Uses NVIDIA unified memory to page optimizer state to
   CPU RAM during memory spikes, preventing out-of-memory crashes.

## How training actually flows

```
base weights stored in 4-bit (NF4), FROZEN
        │  dequantize on the fly to bf16 for each matmul
        ▼
   forward / backward  ──►  gradients flow ONLY into the
                            16-bit LoRA adapters (A, B)
```

The heavy base model sits in 4-bit and never updates. Only the small LoRA
adapters compute in 16-bit and receive gradients. You get LoRA's tiny training
cost **plus** a 4× smaller resident model.

## The config you'll use (BitsAndBytes)

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",            # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16, # compute in bf16
    bnb_4bit_use_double_quant=True,        # double quantization
)
```

Then load the model with `quantization_config=bnb_config` and wrap it with
`prepare_model_for_kbit_training()` before adding LoRA. The notebooks
(`03_qlora_mistral_peft.ipynb`) show the full sequence.

## QLoRA vs LoRA — when to pick which

| | LoRA | QLoRA |
|--|------|-------|
| Base model precision | 16-bit | 4-bit |
| Memory | medium | lowest |
| Speed | faster | a bit slower (dequant overhead) |
| Quality | baseline | ~equal in practice |
| Best for | you have a 24GB+ GPU | free Colab / tight memory |

## Common gotchas

- **`bitsandbytes` needs a CUDA GPU.** It will not run 4-bit on CPU/Mac. Use
  Colab or a cloud GPU for QLoRA.
- Always call `prepare_model_for_kbit_training(model)` — it enables gradient
  checkpointing and fixes layer norms for stable 4-bit training.
- Set `bnb_4bit_compute_dtype=bf16` on Ampere+ GPUs; use fp16 on older cards.
- After training you have a 16-bit adapter. To deploy a single merged model you
  must reload the base in 16-bit, then merge (see `src/merge_adapter.py`).

## Interview soundbite

> "QLoRA freezes the base model in 4-bit NF4 and trains 16-bit LoRA adapters on
> top, using double quantization and paged optimizers. It cuts memory ~4× versus
> LoRA with almost no quality loss, so you can fine-tune a 7B model on a 16GB GPU."

➡️ Next: [06_data_preparation.md](06_data_preparation.md)
