# Deploy with FastAPI + Docker

Serve your fine-tuned model as a REST API.

## 0. Prepare a model

Merge your adapter into a standalone model first (simplest to serve):

```bash
python ../../src/merge_adapter.py \
    --base mistralai/Mistral-7B-v0.1 \
    --adapter ../../results/qlora-adapter \
    --out merged_model
```

(You can also skip merging and set `ADAPTER_PATH` to serve base + adapter.)

## 1. Run locally (no Docker)

```bash
pip install -r requirements.txt
MODEL_PATH=merged_model uvicorn app:app --host 0.0.0.0 --port 8000
```

Test it:

```bash
curl -X POST localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"instruction": "Explain what LoRA is in one sentence."}'
```

Interactive API docs are auto-generated at <http://localhost:8000/docs>.

## 2. Run with Docker

```bash
# Put your merged model where the Dockerfile expects it, then:
docker build -t finetuned-llm-api .
docker run -p 8000:8000 -v $(pwd)/merged_model:/app/merged_model finetuned-llm-api
```

Mounting the model with `-v` keeps the image small. To bake the model into the
image instead, uncomment the `COPY merged_model` line in the Dockerfile.

## 3. GPU notes

- Base the image on `nvidia/cuda:12.1.0-runtime-ubuntu22.04` and install a CUDA
  build of PyTorch.
- Run with `--gpus all`.
- For high throughput, consider **vLLM** or **Text Generation Inference (TGI)**
  instead of this simple server — they add batching and paged attention.

## Endpoints

| Method | Path | Body | Returns |
|--------|------|------|---------|
| GET | `/health` | – | status + which model is loaded |
| POST | `/generate` | `{instruction, max_new_tokens?, temperature?, top_p?}` | `{response}` |
