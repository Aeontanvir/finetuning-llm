"""Gradio chat demo for Hugging Face Spaces.

Drop this file (plus requirements.txt) into a new HF Space (SDK: Gradio) and set
MODEL_ID to your model on the Hub. Reviewers get a browser chat UI — perfect for
a portfolio or interview demo.

Locally:
    MODEL_ID=your-username/my-finetuned-llm python app.py
"""
from __future__ import annotations

import os

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Your fine-tuned model on the Hub (push it with src/push_to_hub.py first).
MODEL_ID = os.environ.get("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)

print(f"Loading {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)
model.eval()


def respond(message: str, history, max_new_tokens: int, temperature: float):
    """Generate a reply for the Gradio ChatInterface."""
    prompt = PROMPT_TEMPLATE.format(instruction=message)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 0.01),
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text.split("### Response:")[-1].strip()


demo = gr.ChatInterface(
    fn=respond,
    title="Fine-tuned LLM Demo",
    description=f"Chatting with `{MODEL_ID}`, fine-tuned with LoRA/QLoRA.",
    additional_inputs=[
        gr.Slider(16, 512, value=256, step=16, label="Max new tokens"),
        gr.Slider(0.0, 1.5, value=0.7, step=0.1, label="Temperature"),
    ],
)

if __name__ == "__main__":
    demo.launch()
