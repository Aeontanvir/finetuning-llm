# Deploy on Hugging Face Hub + Spaces

Two steps: (1) publish your model to the **Hub**, (2) host a live demo on a
**Space**. Great for sharing in interviews — reviewers chat with it in-browser.

## 1. Push your model to the Hub

```bash
export HF_TOKEN=hf_xxx        # create at huggingface.co/settings/tokens
python ../../src/push_to_hub.py \
    --path ../../results/qlora-adapter \
    --repo your-username/my-finetuned-llm
```

You can push either the **adapter** (small) or a **merged** model. For the Space
below, pushing a merged model is simplest because it loads with a single line.

## 2. Create the Space

1. Go to <https://huggingface.co/new-space>.
2. Choose **SDK: Gradio**, pick CPU (free) or a GPU tier.
3. Upload `app.py` and `requirements.txt` from this folder (or `git push` them
   to the Space repo).
4. In the Space **Settings → Variables**, set `MODEL_ID` to your Hub repo, e.g.
   `your-username/my-finetuned-llm`.
5. The Space builds and launches the chat UI automatically.

## Run the demo locally first

```bash
pip install -r requirements.txt
MODEL_ID=your-username/my-finetuned-llm python app.py
# open the printed http://127.0.0.1:7860 URL
```

## Notes

- **Free CPU Spaces** can run small models (≤~1–3B) or merged 7B slowly. For a
  snappy 7B demo, use a GPU Space tier.
- To serve **base + adapter** instead of a merged model, load the base then
  `PeftModel.from_pretrained(base, adapter_repo)` in `app.py`.
- Add a `README.md` with Space front-matter (title, emoji, sdk) if you want a
  nicer Space card; HF generates a default one otherwise.
