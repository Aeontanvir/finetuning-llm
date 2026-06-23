# Deploy locally with Ollama (GGUF)

Run your fine-tuned model on your own laptop, offline, with one command. This
path converts the model to **GGUF** (the llama.cpp format) and runs it via
[Ollama](https://ollama.com).

## Overview

```
merged model (HF format)  ──convert──►  model.gguf  ──quantize──►  model-q4_k_m.gguf
                                                                         │
                                                  Modelfile  ──ollama create──► run
```

## 1. Merge your adapter first

GGUF conversion needs a full model, not an adapter:

```bash
python ../../src/merge_adapter.py \
    --base mistralai/Mistral-7B-v0.1 \
    --adapter ../../results/qlora-adapter \
    --out merged_model
```

## 2. Convert to GGUF with llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && pip install -r requirements.txt

# Convert HF model -> GGUF (FP16)
python convert_hf_to_gguf.py ../merged_model --outfile model-f16.gguf --outtype f16

# Quantize to 4-bit (smaller, faster, runs on CPU/laptop)
./llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M
```

Common quant levels: `Q4_K_M` (best size/quality balance), `Q5_K_M` (a bit
better quality), `Q8_0` (near-lossless, larger).

> Note: llama.cpp's exact script names change over time. If `convert_hf_to_gguf.py`
> isn't present, check the repo for the current converter (older: `convert.py`).

## 3. Create and run the Ollama model

Put `model-q4_k_m.gguf` next to the [Modelfile](Modelfile), then:

```bash
ollama create my-llm -f Modelfile
ollama run my-llm "Explain what LoRA is in one sentence."
```

## 4. Use it from code

Ollama exposes a local API on port 11434:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "my-llm",
  "prompt": "Explain what QLoRA is.",
  "stream": false
}'
```

## Tips

- Edit the `TEMPLATE` in the Modelfile to **exactly match** your training prompt
  format, or quality drops.
- Add `PARAMETER stop` lines for any markers your model uses so it stops cleanly.
- A quantized 7B model needs roughly 4–6 GB of RAM at Q4_K_M.
