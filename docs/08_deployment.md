# 8. Deployment Overview

A fine-tuned model is only useful when something can call it. This course ships
three deployment paths, each in `deployment/`. Pick based on your goal.

## First: adapter or merged model?

After LoRA/QLoRA you have a small **adapter**. You can deploy in two ways:

1. **Base + adapter at runtime** — load the base model, then apply the adapter.
   Flexible (swap adapters), but needs the `peft` runtime and a bit more code.
2. **Merged model** — fold the adapter into the base weights once
   (`src/merge_adapter.py`) to get a normal standalone model. Simpler to serve,
   slightly faster, but larger files and no swapping.

For all three deployment targets below, a **merged** model is the easiest path.

## Path A — FastAPI + Docker (your own REST API)

**When:** you want full control, to run on your own server/cloud, and to expose
a `/generate` endpoint other apps can call.

- `deployment/fastapi/app.py` — FastAPI server loading the model once at startup.
- `deployment/fastapi/Dockerfile` — containerize for reproducible deploys.
- Scales behind a load balancer; add GPU base images for speed.

➡️ [deployment/fastapi/README.md](../deployment/fastapi/README.md)

## Path B — Hugging Face Hub + Spaces (shareable demo)

**When:** you want a public (or private) live demo and an easy way to share/host
weights, great for portfolios and interviews.

- Push your adapter/model to the **Hub** with `src/push_to_hub.py`.
- `deployment/huggingface_spaces/app.py` — a Gradio chat UI that runs on a free
  Space. Reviewers can try your model in the browser.

➡️ [deployment/huggingface_spaces/README.md](../deployment/huggingface_spaces/README.md)

## Path C — Ollama / local GGUF (run on your laptop)

**When:** you want the model to run locally, offline, with a single command, and
low resource use via quantized **GGUF** files.

- Convert the merged model to GGUF with `llama.cpp`, quantize (e.g., Q4_K_M).
- Write a `Modelfile` and `ollama create` your model, then `ollama run`.

➡️ [deployment/ollama/README.md](../deployment/ollama/README.md)

## Choosing quickly

| Goal | Path |
|------|------|
| Programmatic API for an app/product | **FastAPI + Docker** |
| Public demo / portfolio / share weights | **Hugging Face Spaces + Hub** |
| Offline, local, laptop-friendly | **Ollama / GGUF** |
| Highest throughput at scale (advanced) | vLLM or TGI (not covered, see notes) |

## Production considerations (interview bonus)

- **Latency vs throughput:** batch requests; consider vLLM/TGI for high QPS.
- **Quantization at inference:** 8-bit/4-bit serving saves memory.
- **Cost:** GPU vs CPU; merged small models can run on CPU for low traffic.
- **Monitoring:** log prompts/outputs (privacy-aware), track latency and errors.
- **Safety:** add input/output filtering; rate limit the API.
- **Versioning:** tag model + adapter versions so you can roll back.

➡️ Next: [09_glossary_interview.md](09_glossary_interview.md)
