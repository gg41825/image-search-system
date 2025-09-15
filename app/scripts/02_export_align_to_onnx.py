import torch
from models.aligned_embedder import AlignedEmbedder

# ------------------------
# Initialize the model
# ------------------------
device = "cpu"  # Use CPU for ONNX export (safer and more portable)
model = AlignedEmbedder(
    text_model_name="bert-base-uncased",
    vision_model_name="facebook/dinov2-base"
).to(device)
model.eval()

# ------------------------
# Create dummy inputs
# ------------------------
# Assumptions:
# - Batch size = 1
# - Sequence length = 16
# - Image size = 224x224
dummy_input_ids = torch.ones((1, 16), dtype=torch.long).to(device)
dummy_attention_mask = torch.ones((1, 16), dtype=torch.long).to(device)
dummy_pixel_values = torch.randn((1, 3, 224, 224)).to(device)

# ------------------------
# Export to ONNX
# ------------------------
onnx_path = "aligned_embedder.onnx"

torch.onnx.export(
    model,
    (dummy_input_ids, dummy_attention_mask, dummy_pixel_values),
    onnx_path,
    export_params=True,       # Store the trained parameter weights
    opset_version=17,         # Use ONNX opset 17 for better compatibility
    do_constant_folding=True, # Simplify the graph where possible
    input_names=["input_ids", "attention_mask", "pixel_values"],
    output_names=["embedding"],
    dynamic_axes={            # Allow variable-length inputs
        "input_ids": {0: "batch_size", 1: "sequence"},
        "attention_mask": {0: "batch_size", 1: "sequence"},
        "pixel_values": {0: "batch_size"},
        "embedding": {0: "batch_size"},
    },
)

print(f"✅ Exported ONNX model saved at {onnx_path}")