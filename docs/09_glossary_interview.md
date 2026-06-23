# 9. Glossary & Interview Q&A

## Glossary

- **Adapter:** the small set of trained LoRA weights added to a frozen base model.
- **Alignment:** making a model's behavior match human intent/values (RLHF, DPO).
- **BitsAndBytes:** library that provides 4-bit/8-bit quantized model loading.
- **Catastrophic forgetting:** loss of general ability after fine-tuning on a narrow task.
- **Causal LM:** a model that predicts the next token from previous tokens.
- **Checkpoint:** a saved snapshot of model weights.
- **Context window:** max tokens the model can process at once.
- **DPO:** Direct Preference Optimization, simpler alternative to RLHF.
- **Epoch:** one full pass over the training data.
- **EOS token:** end-of-sequence marker; teaches the model when to stop.
- **Gradient accumulation:** simulate a large batch by summing gradients over
  several small batches before updating — saves memory.
- **Gradient checkpointing:** trade compute for memory by recomputing
  activations in the backward pass instead of storing them.
- **LoRA:** Low-Rank Adaptation; trains small low-rank matrices instead of full weights.
- **NF4:** 4-bit NormalFloat datatype used by QLoRA.
- **PEFT:** Parameter-Efficient Fine-Tuning (the family LoRA belongs to).
- **Perplexity:** exp(loss); how "surprised" the model is — lower is better.
- **Quantization:** storing weights in fewer bits (FP16 → INT8/INT4).
- **QLoRA:** LoRA on top of a 4-bit quantized frozen base model.
- **RAG:** Retrieval-Augmented Generation; inject documents into the prompt.
- **Rank (r):** the inner dimension of LoRA matrices; controls capacity.
- **RLHF:** Reinforcement Learning from Human Feedback.
- **SFT:** Supervised Fine-Tuning on input→output pairs.
- **SFTTrainer:** TRL class that handles SFT end to end.
- **Tokenizer:** converts text to/from tokens; must match the model.
- **TRL:** Hugging Face "Transformer Reinforcement Learning" library (SFT, DPO, PPO).

## Interview Q&A

**Q: What's the difference between fine-tuning and RAG?**
Fine-tuning changes the model's weights to teach behavior, style, or skills. RAG
leaves weights alone and supplies knowledge at query time. Use RAG for facts
that change; fine-tuning for consistent behavior/format. They're complementary.

**Q: Explain LoRA in one minute.**
Fine-tuning's weight change ΔW is approximately low-rank, so instead of learning
the full matrix we learn two skinny matrices B and A whose product approximates
ΔW. We freeze the original weights and train only A and B — about 10,000× fewer
parameters. Output is `Wx + (alpha/r)·BAx`. We get tiny, swappable adapters and
almost no catastrophic forgetting.

**Q: What does QLoRA add over LoRA?**
It loads the frozen base model in 4-bit NF4 (with double quantization and paged
optimizers) while training 16-bit LoRA adapters. Memory drops ~4× with little
quality loss, so a 7B model fine-tunes on a 16GB GPU.

**Q: How do you pick r and alpha?**
Start r=16, alpha=32, target the attention projections. Raise r if underfitting;
alpha scales the adapter's contribution (often r or 2r). Validate with held-out loss.

**Q: How do you know fine-tuning worked?**
Train vs. validation loss curves, task metrics (F1/ROUGE/etc.), qualitative
review on fixed prompts, LLM-as-judge, and a regression check on general prompts.

**Q: Your model never stops generating. Why?**
Missing EOS token in the training examples, or a prompt-template mismatch between
training and inference.

**Q: Full fine-tuning vs PEFT — when full?**
Full FT when you have abundant compute/data and a large domain shift, and can
afford a full model copy per task. Otherwise PEFT (LoRA/QLoRA) is the default.

**Q: How do you deploy a LoRA model?**
Either serve base+adapter with `peft`, or merge the adapter into the base and
serve the merged model via FastAPI/vLLM, push to HF Hub/Spaces, or convert to
GGUF for Ollama.

**Q: How do you prevent catastrophic forgetting?**
Prefer PEFT (frozen base), use a low learning rate, fewer epochs, mix in some
general data, and run regression tests on general tasks.

**Q: What hyperparameters matter most for SFT?**
Learning rate (most important), epochs, batch size (with gradient accumulation),
LoRA r/alpha, and max sequence length.

**Q: Memory-saving tricks for limited GPUs?**
4-bit loading (QLoRA), gradient accumulation, gradient checkpointing, paged
optimizer (paged_adamw_8bit), shorter max_seq_length, smaller batch size.

➡️ Back to [README](../README.md) · Start coding in the [notebooks](../notebooks).
