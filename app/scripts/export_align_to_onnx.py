import torch
from models.aligned_embedder import AlignedEmbedder
import os

# ------------------------
# Initialize the model
# ------------------------
device = "cpu"
model = AlignedEmbedder(
    text_model_name="bert-base-uncased",
    vision_model_name="facebook/dinov2-base"
).to(device)
model.eval()

# ------------------------
# Create dummy inputs
# ------------------------
dummy_input_ids = torch.ones((1, 16), dtype=torch.long).to(device)
dummy_attention_mask = torch.ones((1, 16), dtype=torch.long).to(device)
dummy_pixel_values = torch.randn((1, 3, 224, 224)).to(device)

# ------------------------
# Export to ONNX
# ------------------------
# Make sure run this script under the root directory
output_dir = os.path.join("model_repository", "aligned", "1")
os.makedirs(output_dir, exist_ok=True)

onnx_path = os.path.join(output_dir, "model.onnx")

torch.onnx.export(
    model,
    (dummy_input_ids, dummy_attention_mask, dummy_pixel_values),
    onnx_path,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input_ids", "attention_mask", "pixel_values"],
    output_names=["embedding"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence"},
        "attention_mask": {0: "batch_size", 1: "sequence"},
        "pixel_values": {0: "batch_size"},
        "embedding": {0: "batch_size"},
    },
)

print(f"✅ Exported ONNX model saved at {onnx_path}")