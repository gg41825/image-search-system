FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set Hugging Face cache directory
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface
ENV HF_HOME=/root/.cache/huggingface
ENV TORCH_HOME=/root/.cache/torch
RUN mkdir -p $TRANSFORMERS_CACHE $TORCH_HOME

# Accept HF_TOKEN as build argument
ARG HF_TOKEN
ENV HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

# Pre-download Hugging Face models
RUN python -c "import os; \
    from transformers import BertTokenizer, BertModel, AutoImageProcessor, AutoModel; \
    token = os.environ.get('HUGGING_FACE_HUB_TOKEN'); \
    print('Downloading BERT...'); \
    BertTokenizer.from_pretrained('bert-base-uncased', token=token); \
    BertModel.from_pretrained('bert-base-uncased', token=token); \
    print('Downloading DINOv2 from HuggingFace...'); \
    AutoImageProcessor.from_pretrained('facebook/dinov2-base', token=token); \
    AutoModel.from_pretrained('facebook/dinov2-base', token=token); \
    print('Models downloaded')"

# Pre-download DINOv2 from torch.hub
RUN python -c "import torch; \
    print('Downloading DINOv2 from torch.hub...'); \
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14'); \
    print('DINOv2 torch.hub downloaded')"

# Copy app code
COPY app/ /app/

# Create necessary directories
RUN mkdir -p /app/data/uploads /app/data/cache

# Cloud Run sets PORT env var, default to 8080
ENV PORT=8080

# Keep cache location consistent for runtime
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface
ENV HF_HOME=/root/.cache/huggingface
ENV TORCH_HOME=/root/.cache/torch

# Start uvicorn server directly
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]