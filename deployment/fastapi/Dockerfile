# CPU image. For GPU, base on nvidia/cuda:12.1.0-runtime-ubuntu22.04 and
# install the matching CUDA build of torch.
FROM python:3.10-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the server code.
COPY app.py .

# The merged model is expected to be mounted or copied to /app/merged_model.
# At build time you can instead COPY your model directory:
#   COPY merged_model ./merged_model
ENV MODEL_PATH=/app/merged_model

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
