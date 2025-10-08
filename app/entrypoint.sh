#!/bin/bash
set -e

#Based on the directory where the script is located (usually /app in the container and ./app locally)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="$SCRIPT_DIR"


# Detect base path. Model repository: Use /model_repository in the container first; otherwise, fall back to ./model_repository in the project root directory
MODEL_ROOT="/model_repository"
if [ ! -d "$MODEL_ROOT" ] && [ -d "$BASE_PATH/../model_repository" ]; then
  MODEL_ROOT="$BASE_PATH/../model_repository"
fi

MODEL_PATH="$MODEL_ROOT/aligned/1/model.onnx"


# Check if ONNX model exists
if [ ! -f "$MODEL_PATH" ]; then
  echo "⚠️ Model not found at: $MODEL_PATH"
  echo "→ Exporting ONNX model..."
  PYTHONPATH="$BASE_PATH" python3 "$BASE_PATH/scripts/export_align_to_onnx.py"
else
  echo "✅ ONNX model already exists."
fi


# Continue to execute the original CMD
exec "$@"