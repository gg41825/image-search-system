import os
import shutil
import argparse
from pathlib import Path
from typing import List
import torch
from transformers import CLIPProcessor, CLIPModel
import warnings

warnings.filterwarnings("ignore")

# ---- Security Settings: Avoid SDPA Export Errors ----
os.environ["PYTORCH_SDP_DISABLE_FLASH_ATTENTION"] = "1"
os.environ["PYTORCH_SDP_DISABLE_MEM_EFFICIENT_ATTENTION"] = "1"
os.environ["PYTORCH_SDP_DISABLE_HEURISTIC_ATTENTION"] = "1"


# ---- Configurations ----
MODEL_NAME: str = "openai/clip-vit-base-patch32"
OUT_DIR: Path = Path("onnx_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OPSET_VERSION: int = 14

DST_BASE: Path = Path("model_repository")

MODELS = {
    "clip_image_encoder.onnx": DST_BASE / "clip_image_encoder/1/model.onnx",
    "clip_text_encoder.onnx": DST_BASE / "clip_text_encoder/1/model.onnx",
}


def export_onnx_model(
    model: torch.nn.Module,
    dummy_inputs: tuple,
    file_name: str,
    input_names: List[str],
    output_names: List[str],
    dynamic_axes: dict,
) -> None:
    """Helper function to export a PyTorch model to ONNX format."""
    out_path = OUT_DIR / file_name
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_inputs,
            str(out_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=OPSET_VERSION,
        )
    print(f"✅ Exported {file_name} → {out_path.resolve()}")


def deploy_models() -> None:
    """Move exported ONNX models to Triton model repository."""
    for src_name, dst_path in MODELS.items():
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(OUT_DIR / src_name), str(dst_path))
        print(f"📦 Deployed {src_name} → {dst_path.resolve()}")


def main(deploy: bool = False) -> None:
    # ---- Load the model and processor ----
    model = CLIPModel.from_pretrained(MODEL_NAME).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    # ---- Dummy inputs ----
    texts = ["a pair of sneakers"]
    images = torch.randn(1, 3, 224, 224)  # Fake image input
    ti = processor(text=texts, return_tensors="pt", padding=True, truncation=True)

    # ---- Export Vision Encoder ----
    export_onnx_model(
        model.vision_model,
        (images,),
        "clip_image_encoder.onnx",
        input_names=["pixel_values"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "last_hidden_state": {0: "batch", 1: "seq"},
            "pooler_output": {0: "batch"},
        },
    )

    # ---- Export Text Encoder ----
    export_onnx_model(
        model.text_model,
        (ti["input_ids"], ti["attention_mask"]),
        "clip_text_encoder.onnx",
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "last_hidden_state": {0: "batch", 1: "seq"},
            "pooler_output": {0: "batch"},
        },
    )

    print("\n🎉 All models exported successfully!")

    if deploy:
        deploy_models()
        print("\n🚀 Models deployed to Triton repository!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export CLIP model to ONNX (and optionally deploy).")
    parser.add_argument("--deploy", action="store_true", help="Move ONNX models to model_repository/ for Triton")
    args = parser.parse_args()

    main(deploy=args.deploy)
