"""Load YAML config files into typed dataclasses.

One place to read all hyperparameters so the training scripts stay clean.
Usage:
    from src.config import load_config
    cfg = load_config("configs/lora.yaml")
    print(cfg.model.base_model)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import yaml


@dataclass
class ModelConfig:
    base_model: str
    trust_remote_code: bool = False


@dataclass
class DataConfig:
    train_file: str
    format: str = "alpaca"          # "alpaca" or "chat"
    max_seq_length: int = 512
    val_split: float = 0.1


@dataclass
class LoraConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


@dataclass
class TrainingConfig:
    output_dir: str = "results/adapter"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    logging_steps: int = 5
    eval_strategy: str = "steps"
    eval_steps: int = 20
    save_strategy: str = "epoch"
    bf16: bool = True
    fp16: bool = False
    optim: str = "adamw_torch"
    gradient_checkpointing: bool = False
    report_to: str = "tensorboard"


@dataclass
class QuantConfig:
    load_in_4bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"


@dataclass
class Config:
    model: ModelConfig
    data: DataConfig
    lora: LoraConfig
    training: TrainingConfig
    quantization: QuantConfig


def load_config(path: str) -> Config:
    """Read a YAML file and build a fully-typed Config object."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    return Config(
        model=ModelConfig(**raw["model"]),
        data=DataConfig(**raw["data"]),
        lora=LoraConfig(**raw["lora"]),
        training=TrainingConfig(**raw["training"]),
        quantization=QuantConfig(**raw.get("quantization", {})),
    )


if __name__ == "__main__":
    import sys

    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/lora.yaml")
    print("Loaded config:")
    print("  base_model     :", cfg.model.base_model)
    print("  train_file     :", cfg.data.train_file)
    print("  lora r / alpha :", cfg.lora.r, "/", cfg.lora.lora_alpha)
    print("  4-bit          :", cfg.quantization.load_in_4bit)
