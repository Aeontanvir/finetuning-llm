"""Fine-tune an LLM with QLoRA (4-bit base model). Requires a CUDA GPU.

Example (on Colab/GPU):
    python src/train_qlora.py --config configs/qlora.yaml

Difference vs train_lora.py: the base model is loaded in 4-bit via BitsAndBytes
and prepared for k-bit training. Everything else is the same LoRA flow.
"""
from __future__ import annotations

import argparse

import torch
from peft import (LoraConfig as PeftLoraConfig, prepare_model_for_kbit_training)
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments)
from trl import SFTTrainer

from src.config import load_config
from src.data_prep import build_dataset


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    if not cfg.quantization.load_in_4bit:
        raise SystemExit("Use configs/qlora.yaml (load_in_4bit: true) for QLoRA.")
    if not torch.cuda.is_available():
        raise SystemExit("QLoRA needs a CUDA GPU (bitsandbytes). Use Colab/cloud.")

    print(f"== QLoRA fine-tuning: {cfg.model.base_model} ==")

    # 1. 4-bit quantization config (NF4 + double quant + bf16 compute).
    compute_dtype = getattr(torch, cfg.quantization.bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.quantization.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.quantization.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # 2. Tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.base_model, trust_remote_code=cfg.model.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3. Load base model in 4-bit and prepare it for k-bit training.
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=cfg.model.trust_remote_code,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg.training.gradient_checkpointing
    )

    # 4. LoRA adapter config.
    peft_config = PeftLoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
        target_modules=cfg.lora.target_modules,
        task_type="CAUSAL_LM",
    )

    # 5. Dataset.
    ds = build_dataset(cfg, tokenizer)
    train_ds = ds["train"] if "train" in ds else ds
    eval_ds = ds["test"] if "test" in ds else None

    # 6. Training args (note paged optimizer from the config).
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

    # 7. Train and save the adapter.
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
    trainer.save_model(cfg.training.output_dir)
    tokenizer.save_pretrained(cfg.training.output_dir)
    print(f"Done. QLoRA adapter saved to {cfg.training.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qlora.yaml")
    main(parser.parse_args().config)
