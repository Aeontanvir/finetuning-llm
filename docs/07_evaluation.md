# 7. Evaluation — Did It Actually Work?

A model that trains without errors is not a model that works. You must measure.

## 1. Watch the loss curves (during training)

- **Training loss** should steadily decrease.
- **Validation loss** should also decrease, then flatten. When validation loss
  starts *rising* while training loss keeps falling, you are **overfitting** —
  stop or reduce epochs / add data / increase dropout.
- Loss that is flat from the start → learning rate too low, or data/template bug.
- Loss that explodes to NaN → learning rate too high, or fp16 instability (try bf16).

Use TensorBoard or Weights & Biases (`report_to="tensorboard"`).

## 2. Quantitative metrics (pick what fits the task)

| Task | Metric |
|------|--------|
| Classification | accuracy, F1 |
| Summarization | ROUGE, BERTScore |
| Translation | BLEU, chrF |
| General LM quality | perplexity (lower = better) |
| Code | pass@k (unit tests pass) |

The `evaluate` library implements most of these:
```python
import evaluate
rouge = evaluate.load("rouge")
rouge.compute(predictions=preds, references=refs)
```

## 3. Qualitative evaluation (do not skip)

Numbers miss a lot. Keep a fixed set of ~20–50 real prompts and **read the
outputs** before and after fine-tuning. Look for: correct format, right tone,
following the instruction, no hallucinated rambling, proper stopping.

## 4. LLM-as-a-judge

Use a strong model (e.g., GPT-4 class) to score your model's answers on a rubric
(helpfulness, correctness, format) 1–5. Scales better than human review and
correlates reasonably well. Be aware of its biases (length, position).

## 5. Check for regressions

Fine-tuning can cause **catastrophic forgetting**. Test a few *general* prompts
unrelated to your task to confirm the model did not get dumber overall. LoRA/QLoRA
reduce this risk because the base weights are frozen.

## 6. Common failure modes and fixes

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Never stops generating | missing EOS token in data | add `eos_token` to each example |
| Ignores instructions | template mismatch train vs infer | use identical prompt template |
| Repetitive / degenerate text | overfit or LR too high | fewer epochs, lower LR, more data |
| No change from base | LR too low, adapter not applied | raise LR, verify `print_trainable_parameters` |
| Great on train, bad on new prompts | overfitting | early stop, more diverse data, dropout |
| NaN loss | fp16 instability / high LR | use bf16, lower LR, grad clipping |

## 7. A note on alignment (RLHF / DPO)

After SFT, you can further align a model to human *preferences*:
- **RLHF** trains a reward model from human rankings, then optimizes with PPO.
  Powerful but complex and unstable.
- **DPO (Direct Preference Optimization)** skips the reward model and optimizes
  directly on preferred-vs-rejected pairs. Simpler, increasingly the default.
  TRL provides a `DPOTrainer`. This is a natural next step after this course.

## Minimal evaluation recipe for this repo

1. Hold out 10% as validation; set `eval_strategy="steps"`.
2. Watch train vs. validation loss in TensorBoard.
3. Run `src/inference.py` on your 20 fixed test prompts, before and after.
4. Eyeball format, instruction-following, and stopping behavior.

➡️ Next: [08_deployment.md](08_deployment.md)
