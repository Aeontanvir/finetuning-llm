"""Run inference with a fine-tuned model (adapter or merged).

Two modes:
  - base + adapter:  --base <name> --adapter results/lora-adapter
  - merged model:    --model merged_model

Example:
    python src/inference.py --base TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
        --adapter results/lora-adapter \\
        --prompt "Explain what LoRA is in one sentence."
"""
from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Must match the template used in training (see src/data_prep.py).
PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


def load(model_path: str | None, base: str | None, adapter: str | None):
    """Return (model, tokenizer) for either a merged model or base+adapter."""
    if model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        return model, tokenizer

    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(adapter or base)
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    return model, tokenizer


def generate(model, tokenizer, instruction: str, max_new_tokens: int = 256) -> str:
    prompt = PROMPT_TEMPLATE.format(instruction=instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text.split("### Response:")[-1].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Path to a merged model directory")
    parser.add_argument("--base", help="Base model name (for adapter mode)")
    parser.add_argument("--adapter", help="Path to a trained adapter")
    parser.add_argument("--prompt", help="Single prompt; omit for interactive chat")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    if not args.model and not args.base:
        raise SystemExit("Provide --model (merged) or --base [--adapter].")

    model, tokenizer = load(args.model, args.base, args.adapter)
    model.eval()

    if args.prompt:
        print(generate(model, tokenizer, args.prompt, args.max_new_tokens))
        return

    print("Interactive mode. Type 'quit' to exit.")
    while True:
        try:
            user = input("\nInstruction> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user.lower() in {"quit", "exit"}:
            break
        if user:
            print(generate(model, tokenizer, user, args.max_new_tokens))


if __name__ == "__main__":
    main()
