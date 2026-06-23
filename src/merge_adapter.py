"""Merge a trained LoRA/QLoRA adapter into the base model weights.

Produces a standalone model directory you can serve without the `peft` runtime
(used by all three deployment paths). Note: reload the base in 16-bit (not 4-bit)
before merging so the merged weights are full precision.

Example:
    python src/merge_adapter.py \\
        --base mistralai/Mistral-7B-v0.1 \\
        --adapter results/qlora-adapter \\
        --out merged_model
"""
from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge(base: str, adapter: str, out: str) -> None:
    print(f"Loading base model in 16-bit: {base}")
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="auto"
    )

    print(f"Applying adapter: {adapter}")
    model = PeftModel.from_pretrained(model, adapter)

    print("Merging adapter into base weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {out}")
    model.save_pretrained(out, safe_serialization=True)

    # Save the tokenizer alongside so the merged dir is self-contained.
    tokenizer = AutoTokenizer.from_pretrained(adapter)
    tokenizer.save_pretrained(out)
    print("Merge complete. The merged model is ready to serve or convert to GGUF.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Base model name or path")
    parser.add_argument("--adapter", required=True, help="Path to trained adapter")
    parser.add_argument("--out", default="merged_model", help="Output directory")
    a = parser.parse_args()
    merge(a.base, a.adapter, a.out)
