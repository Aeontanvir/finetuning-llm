"""Fine-tune an LLM with LoRA (16-bit base model).

Example:
    python src/train_lora.py --config configs/lora.yaml

This is a thin, readable wrapper around Hugging Face TRL's SFTTrainer. The
heavy lifting (data formatting) lives in src/data_prep.py and all knobs live in
the YAML config so you never edit this file to change a run.
"""
from __future__ import annotations

import argparse

import torch
from peft import LoraConfig as PeftLoraConfig
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          TrainingArguments)
from trl import SFTTrainer

from src.config import load_config
from src.data_prep import build_dataset


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    print(f"== LoRA fine-tuning: {cfg.model.base_model} ==")

    # 1. Tokenizer. A pad token is required for batching; reuse EOS if missing.
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.base_model, trust_remote_code=cfg.model.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Base model in full/half precision (no quantization for plain LoRA).
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model,
        torch_dtype=torch.bfloat16 if cfg.training.bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=cfg.model.trust_remote_code,
    )
    model.config.use_cache = False  # required with gradient checkpointing

    # 3. LoRA adapter config — only these small matrices will train.
    peft_config = PeftLoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
        target_modules=cfg.lora.target_modules,
        task_type="CAUSAL_LM",
    )

    # 4. Dataset (returns a DatasetDict with train/test when val_split > 0).
    ds = build_dataset(cfg, tokenizer)
    train_ds = ds["train"] if "train" in ds else ds
    eval_ds = ds["test"] if "test" in ds else None

    # 5. Training arguments straight from the config.
    args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        warmup_ratio=cfg.training.warmup_ratio,
        logging_steps=cfg.training.logging_steps,
        eval_strategy=cfg.training.eval_strategy if eval_ds else "no",
        eval_steps=cfg.training.eval_steps,
        save_strategy=cfg.training.save_strategy,
        bf16=cfg.training.bf16,
        fp16=cfg.training.fp16,
        optim=cfg.training.optim,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        report_to=cfg.training.report_to,
    )

    # 6. Train. SFTTrainer applies the LoRA adapter and trains on "text".
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=cfg.data.max_seq_length,
        tokenizer=tokenizer,
    )

    trainer.train()

    # 7. Save the adapter (small) + tokenizer.
    trainer.save_model(cfg.training.output_dir)
    tokenizer.save_pretrained(cfg.training.output_dir)
    print(f"Done. Adapter saved to {cfg.training.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lora.yaml")
    main(parser.parse_args().config)
