"""FastAPI server that serves a fine-tuned LLM behind a /generate endpoint.

The model is loaded once at startup (slow) and reused for every request (fast).
Configure via environment variables:
    MODEL_PATH   path to a merged model OR a base model name
    ADAPTER_PATH (optional) path to a LoRA adapter to apply on top of MODEL_PATH

Run locally:
    MODEL_PATH=merged_model uvicorn app:app --host 0.0.0.0 --port 8000

Then:
    curl -X POST localhost:8000/generate -H 'Content-Type: application/json' \\
         -d '{"instruction": "Explain LoRA in one sentence."}'
"""
from __future__ import annotations

import os

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = os.environ.get("MODEL_PATH", "merged_model")
ADAPTER_PATH = os.environ.get("ADAPTER_PATH")  # optional

PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)

app = FastAPI(title="Fine-tuned LLM API")

# Globals populated on startup.
model = None
tokenizer = None


class GenerateRequest(BaseModel):
    instruction: str
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9


class GenerateResponse(BaseModel):
    response: str


@app.on_event("startup")
def load_model() -> None:
    """Load the model + tokenizer a single time when the server boots."""
    global model, tokenizer
    print(f"Loading model from {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH or MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if ADAPTER_PATH:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()
    print("Model loaded.")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_PATH, "adapter": ADAPTER_PATH}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    prompt = PROMPT_TEMPLATE.format(instruction=req.instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            do_sample=True,
            temperature=req.temperature,
            top_p=req.top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    answer = text.split("### Response:")[-1].strip()
    return GenerateResponse(response=answer)
