"""Load and format a fine-tuning dataset.

Supports two layouts:
  - "alpaca": {"instruction", "input", "output"}
  - "chat":   {"messages": [{"role", "content"}, ...]}

Produces a `datasets.Dataset` with a single "text" column, fully formatted with
the right prompt template and ending in the EOS token (so the model learns to
stop). The training scripts pass this to TRL's SFTTrainer.

Run standalone to preview the formatting:
    python src/data_prep.py --config configs/lora.yaml --preview
"""
from __future__ import annotations

import argparse
from typing import Dict

from datasets import Dataset, load_dataset

ALPACA_TEMPLATE_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

ALPACA_TEMPLATE_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{output}"
)


def format_alpaca(example: Dict, eos: str) -> Dict:
    """Render one alpaca-style row into a single training string."""
    if example.get("input"):
        text = ALPACA_TEMPLATE_WITH_INPUT.format(
            instruction=example["instruction"],
            input=example["input"],
            output=example["output"],
        )
    else:
        text = ALPACA_TEMPLATE_NO_INPUT.format(
            instruction=example["instruction"],
            output=example["output"],
        )
    return {"text": text + eos}


def format_chat(example: Dict, tokenizer) -> Dict:
    """Render a chat-style row using the tokenizer's own chat template."""
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return {"text": text}


def build_dataset(cfg, tokenizer) -> Dataset:
    """Load the raw file and return a formatted, split dataset dict."""
    raw = load_dataset("json", data_files=cfg.data.train_file, split="train")
    eos = tokenizer.eos_token or "</s>"

    if cfg.data.format == "alpaca":
        formatted = raw.map(lambda ex: format_alpaca(ex, eos),
                            remove_columns=raw.column_names)
    elif cfg.data.format == "chat":
        formatted = raw.map(lambda ex: format_chat(ex, tokenizer),
                            remove_columns=raw.column_names)
    else:
        raise ValueError(f"Unknown data format: {cfg.data.format}")

    if cfg.data.val_split and cfg.data.val_split > 0:
        return formatted.train_test_split(test_size=cfg.data.val_split, seed=42)
    return formatted


def _preview(config_path: str) -> None:
    """Print a couple of formatted examples without needing the model weights."""
    from src.config import load_config

    cfg = load_config(config_path)
    raw = load_dataset("json", data_files=cfg.data.train_file, split="train")
    print(f"Loaded {len(raw)} examples from {cfg.data.train_file}\n")
    for i, ex in enumerate(raw.select(range(min(2, len(raw))))):
        out = format_alpaca(ex, eos="</s>")
        print(f"--- Example {i} ---\n{out['text']}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lora.yaml")
    parser.add_argument("--preview", action="store_true",
                        help="Print formatted examples and exit.")
    args = parser.parse_args()
    if args.preview:
        _preview(args.config)
    else:
        print("Use --preview to inspect formatting, or import build_dataset().")
