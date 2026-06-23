# 2. Transformer & LLM Basics You Must Know

You do not need to derive attention from scratch, but you must be able to explain
these ideas clearly. They directly affect every fine-tuning decision you make.

## 1. Tokens, not words

LLMs do not see text — they see **tokens** (sub-word chunks). "fine-tuning"
might be 3 tokens. The component that does this is the **tokenizer**, and it is
tied to the model. **Always load the tokenizer that matches your model.**

Why it matters for fine-tuning:
- Your dataset is measured in tokens, not characters. Costs and `max_length`
  limits are token-based.
- Padding, truncation, and the special tokens (`<s>`, `</s>`, `[PAD]`) come from
  the tokenizer and must be set correctly or training silently breaks.

## 2. The Transformer in one paragraph

A decoder-only LLM (GPT, Llama, Mistral) is a stack of identical **transformer
blocks**. Each block has two key parts: **self-attention** (every token looks at
previous tokens to gather context) and a **feed-forward network (MLP)**. Inputs
are turned into vectors (**embeddings**), passed through the stack, and the final
layer predicts the probability of the next token. That is it — an LLM is a very
good **next-token predictor**.

## 3. Causal language modeling (the training objective)

During training the model is shown a sequence and learns to predict each next
token given everything before it. The loss is **cross-entropy** between the
predicted token distribution and the actual next token. Fine-tuning uses the
exact same objective — you are just changing *which* text it predicts on.

## 4. Weights, parameters, and why size matters

- A "7B model" has 7 billion **parameters** (the numbers we adjust).
- Each parameter in full precision (FP32) is 4 bytes → 7B × 4 = **28 GB** just
  to store the weights. Add gradients + optimizer state and full fine-tuning a
  7B model needs **>80 GB** of GPU memory.
- This memory wall is the entire reason **LoRA** and **QLoRA** exist (docs 4–5).

## 5. Precision / dtype (FP32, FP16, BF16, INT8, INT4)

Fewer bits = less memory and faster math, at some cost to accuracy.

| dtype | bytes/param | typical use |
|-------|-------------|-------------|
| FP32  | 4 | reference precision, small models |
| FP16 / BF16 | 2 | standard for training big models (mixed precision) |
| INT8  | 1 | inference, 8-bit loading |
| INT4 (NF4) | 0.5 | QLoRA training, 4-bit loading |

BF16 is usually preferred over FP16 for stability if your GPU supports it (Ampere+).

## 6. The forward pass, backward pass, and what gets stored

- **Forward pass:** compute the prediction and the loss.
- **Backward pass:** compute **gradients** (how to change each weight to reduce loss).
- **Optimizer step:** Adam/AdamW updates weights using gradients + momentum.

Memory is dominated by: weights + gradients + optimizer state + activations.
LoRA shrinks the gradient/optimizer part; quantization shrinks the weights part.

## 7. Context window

The maximum number of tokens the model can attend to at once (e.g., 4k, 8k,
32k). Your training examples must fit inside it (`max_seq_length`). Longer
context = more memory.

## Key terms cheat sheet

- **Base model vs. instruct model:** a base model only completes text; an
  *instruct/chat* model has already been instruction-tuned to follow prompts.
  Fine-tune a base model when you want full control; fine-tune an instruct model
  to adjust an already-helpful model.
- **Checkpoint:** a saved snapshot of weights during/after training.
- **Epoch:** one full pass over your dataset.
- **Batch size:** examples processed before one weight update.

➡️ Next: [03_finetuning_methods.md](03_finetuning_methods.md)
