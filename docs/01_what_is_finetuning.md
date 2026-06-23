# 1. What Is Fine-Tuning (and When Should You Use It)?

## The one-sentence definition

Fine-tuning is the process of taking a model that was already trained on a huge
general corpus (a *pre-trained* model) and continuing to train it on a smaller,
task- or domain-specific dataset so it behaves the way you want.

## Why not just train from scratch?

Training a base LLM costs millions of dollars and needs trillions of tokens.
You almost never want to do that. Instead you stand on the shoulders of a
pre-trained model that already understands language, facts, and reasoning, and
you nudge it toward your specific goal.

## The four ways to adapt an LLM (know this for interviews)

There is a spectrum of effort. Pick the *least* expensive option that solves
your problem.

1. **Prompt engineering** — just write better prompts. No training. Free and
   instant. Try this first.
2. **Retrieval-Augmented Generation (RAG)** — inject relevant documents into the
   prompt at query time. Best when the problem is *knowledge* the model lacks.
3. **Fine-tuning** — change the model's weights. Best when the problem is
   *behavior, format, tone, or a skill* that prompting cannot reliably produce.
4. **Pre-training / continued pre-training** — teach the model a new language or
   domain from large unlabeled text. Rare and expensive.

> **Interview trap:** "My chatbot gives wrong facts about our 2025 product."
> That is a *knowledge* gap → use **RAG**, not fine-tuning. Fine-tuning teaches
> *style and skills*, not fresh facts. Mixing these up is the most common mistake.

## Good reasons to fine-tune

- You need a **consistent output format** (e.g., always valid JSON, a specific
  tone, a fixed report structure).
- You have a **narrow, repeated task** (classification, summarization in your
  house style, SQL generation for your schema).
- You want a **smaller, cheaper model** to match a larger one on your task.
- You need **lower latency / no giant prompt** — the behavior is baked in.
- Your data is **private** and cannot go into a third-party prompt.

## When NOT to fine-tune

- The knowledge changes often → use RAG.
- You have very little data (<a few hundred good examples) and prompting works.
- You have not yet tried prompt engineering seriously.

## Types of fine-tuning by *objective*

- **Supervised Fine-Tuning (SFT):** train on (input → desired output) pairs.
  This is 90% of what people mean by "fine-tuning" and what this course covers.
- **Instruction tuning:** a flavor of SFT where data is (instruction → response),
  teaching the model to follow commands.
- **Preference tuning (RLHF / DPO):** align the model to human preferences using
  ranked pairs. Advanced; covered conceptually in doc 7.

## Mental model

```
Pre-trained base model  ──(your small labeled dataset)──►  Fine-tuned model
   knows language,                continued training            does YOUR task
   general knowledge                                            in YOUR style
```

➡️ Next: [02_llm_basics.md](02_llm_basics.md) — the minimum architecture
knowledge you need before touching the code.
