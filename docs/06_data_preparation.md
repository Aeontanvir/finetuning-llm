# 6. Data Preparation & Prompt Formatting

Garbage in, garbage out. Data quality matters more than model size for
fine-tuning. This is where most projects succeed or fail.

## How much data do I need?

- **Style/format tasks:** 100–1,000 high-quality examples can be plenty.
- **New skill / harder task:** a few thousand to tens of thousands.
- **Quality > quantity.** 500 clean, consistent examples beat 5,000 noisy ones.

## The shape of fine-tuning data

Most SFT datasets are **instruction → response** pairs, often with an optional
input/context. Three common JSON layouts:

**Alpaca style**
```json
{"instruction": "Summarize the text.", "input": "Long text...", "output": "Short summary."}
```

**Chat / messages style** (best for chat models)
```json
{"messages": [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "What is LoRA?"},
  {"role": "assistant", "content": "LoRA is a parameter-efficient..."}
]}
```

**Plain prompt/completion**
```json
{"prompt": "Translate to French: Hello", "completion": " Bonjour"}
```

Use **JSONL** (one JSON object per line) — it streams well and `datasets` loads
it directly.

## Prompt templates: the #1 cause of bad results

The model must see the **same template** at training time and at inference time.
If you train with `### Instruction:` headers but query with plain text, quality
collapses.

A simple, reliable Alpaca-style template:

```
Below is an instruction that describes a task. Write a response that
appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

For **chat models**, do not invent your own template — use the tokenizer's built
in one:

```python
text = tokenizer.apply_chat_template(messages, tokenize=False)
```

This inserts the exact special tokens (e.g., `[INST]`, `<|user|>`) the model was
trained on. `src/data_prep.py` handles both styles for you.

## The EOS token (a classic bug)

Each training example must end with the end-of-sequence token (`</s>` /
`tokenizer.eos_token`). If you forget it, your fine-tuned model **never stops
generating** — it rambles forever. TRL's `SFTTrainer` and our `data_prep.py`
add it for you, but always verify.

## Train on completions only (optional but powerful)

By default the model learns to predict every token, including the prompt. Often
you only want it to learn the *response*. You can **mask the prompt tokens** in
the labels (set them to `-100`) so loss is computed only on the answer. TRL
supports this via `DataCollatorForCompletionOnlyLM`.

## Splitting your data

Always hold out a **validation set** (~5–10%) you never train on, so you can
detect overfitting (doc 7). Keep a small **test set** of real prompts for a
final human eyeball check.

## Cleaning checklist

- Remove duplicates and near-duplicates.
- Fix inconsistent formatting (the model will copy your inconsistencies).
- Drop examples longer than `max_seq_length` or truncate sensibly.
- Balance classes/topics if one dominates.
- Spot-check 20 random examples by hand. Always.

## Tokenization parameters

- `max_seq_length` (e.g., 512, 1024, 2048): longer = more memory.
- `padding`: pad to longest in batch (dynamic) to save compute.
- `truncation=True`: required so over-long examples don't crash training.

See it in action: [../notebooks/04_finetune_llama2.ipynb](../notebooks/04_finetune_llama2.ipynb)
and `src/data_prep.py`.

➡️ Next: [07_evaluation.md](07_evaluation.md)
