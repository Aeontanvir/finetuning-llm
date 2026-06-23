# Fine-Tuning LLMs — A Step-by-Step Mastery Guide

A complete, hands-on course for fine-tuning Large Language Models — from the
theory you need in an interview, to runnable code, to **deploying** your
fine-tuned model in three different ways.

> Repo: [Aeontanvir/finetuning-llm](https://github.com/Aeontanvir/finetuning-llm)

---

## Who this is for

- You want to **understand** fine-tuning well enough to explain it in an interview.
- You want **working code** you can run, modify, and reuse later.
- You want to take a model all the way to **deployment** (API, demo, and local).

No prior fine-tuning experience required. Python and basic ML are assumed.

---

## Learning path

Work through these in order. Each doc is self-contained and links to the code.

| # | Topic | Doc |
|---|-------|-----|
| 1 | What fine-tuning is and when to use it | [docs/01_what_is_finetuning.md](docs/01_what_is_finetuning.md) |
| 2 | Transformer & LLM basics you must know | [docs/02_llm_basics.md](docs/02_llm_basics.md) |
| 3 | Fine-tuning methods overview (full / LoRA / QLoRA / PEFT) | [docs/03_finetuning_methods.md](docs/03_finetuning_methods.md) |
| 4 | LoRA explained from scratch | [docs/04_lora_explained.md](docs/04_lora_explained.md) |
| 5 | QLoRA & quantization explained | [docs/05_qlora_explained.md](docs/05_qlora_explained.md) |
| 6 | Data preparation & prompt formatting | [docs/06_data_preparation.md](docs/06_data_preparation.md) |
| 7 | Evaluation: did it actually work? | [docs/07_evaluation.md](docs/07_evaluation.md) |
| 8 | Deployment overview | [docs/08_deployment.md](docs/08_deployment.md) |
| 9 | Glossary & interview Q&A | [docs/09_glossary_interview.md](docs/09_glossary_interview.md) |

## Notebooks (learn by running)

| Notebook | What you build |
|----------|----------------|
| [01_full_finetuning_basics.ipynb](notebooks/01_full_finetuning_basics.ipynb) | Full fine-tuning of a small model, the foundation |
| [02_lora_finetuning.ipynb](notebooks/02_lora_finetuning.ipynb) | Parameter-efficient fine-tuning with LoRA |
| [03_qlora_mistral_peft.ipynb](notebooks/03_qlora_mistral_peft.ipynb) | 4-bit QLoRA on Mistral with PEFT |
| [04_finetune_llama2.ipynb](notebooks/04_finetune_llama2.ipynb) | Instruction-tuning Llama 2 with TRL/SFTTrainer |

## Production scripts (`src/`)

Reusable, configurable Python instead of notebook cells:

- `config.py` — one place for all hyperparameters (YAML-driven)
- `data_prep.py` — load, format, and tokenize datasets
- `train_lora.py` / `train_qlora.py` — CLI training entry points
- `merge_adapter.py` — merge LoRA adapter into base weights
- `inference.py` — load and chat with your fine-tuned model
- `push_to_hub.py` — publish adapter/model to the Hugging Face Hub

## Deployment (`deployment/`)

| Path | What you get |
|------|--------------|
| [fastapi/](deployment/fastapi) | REST API + Dockerfile to serve your model |
| [huggingface_spaces/](deployment/huggingface_spaces) | Gradio demo for Hugging Face Spaces |
| [ollama/](deployment/ollama) | Convert to GGUF and run locally with Ollama |

---

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/Aeontanvir/finetuning-llm.git
cd finetuning-llm
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. (Optional) prepare a sample dataset
python src/data_prep.py --preview

# 3. Fine-tune with LoRA
python src/train_lora.py --config configs/lora.yaml

# 4. Chat with your model
python src/inference.py --adapter ./results/lora-adapter
```

> **Hardware note:** LoRA/QLoRA on a 7B model needs a GPU (Google Colab T4 is
> enough for QLoRA). The small-model examples in notebook 01 run on CPU.

---

## How this repo is built

Every concept and file lands as its own small, readable git commit, so you can
follow the history like a tutorial. See `git log --oneline`.

## License

MIT — see [LICENSE](LICENSE).
