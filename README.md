# Product Scraper

This script scrapes product data from [Bergfreunde.de](https://www.bergfreunde.de) for a list of predefined categories.  
It extracts product **brand, title, category, image URL, and price** from each category page.

---

## Features
- Crawls multiple product categories (defined in `categories_map`)
- Extracts:
  - Brand name
  - Product title
  - Category
  - Image URL
  - Price (converted to float, e.g., `"€ 54,97"` → `54.97`)
- Saves results into a `app/data/products.json` file
- Adds delays between requests to avoid overloading the server

## Usage
1. Scrape Product Data

First, run the crawler script to collect product information and images:

python scripts/01_get_image_data.py


This will create a products.json file in the app/data.

Example entry in products.json:

```
{
  "_id": "1",
  "name": "Patagonia - Worn Wear Patch Kit Reparaturset",
  "category": "Accessoires",
  "price": 24.95,
  "image_url": "https://www.bfgcdn.com/out/pictures/generated/product/1/215_215_90/sol_105-0503-0911_pic1_1.jpg"
}
```
---

# CLIP ONNX Exporter

This script exports [OpenAI CLIP](https://huggingface.co/openai/clip-vit-base-patch32) models
to ONNX format, and optionally deploys them into a [Triton Inference Server](https://github.com/triton-inference-server/server) `model_repository`.

---

## 📦 Requirements

- Python 3.9+
- PyTorch 2.x
- Hugging Face Transformers
- ONNX (>=1.14.1 for opset 14, >=1.15.0 for opset 17)
- (Optional) [onnxruntime](https://github.com/microsoft/onnxruntime) to validate ONNX models


## 🚀 Usage

Run the script to export both the vision encoder and text encoder:
```
python scripts/export_clip_to_onnx.py
```

By default, the ONNX models will be saved under:
```
onnx_out/
├── clip_image_encoder.onnx
└── clip_text_encoder.onnx
```
Export + Deploy to Triton

Use the --deploy flag to automatically move the exported ONNX models into the proper Triton structure:
```
python scripts/export_clip_to_onnx.py --deploy
```

Resulting structure:
```
model_repository/
├── clip_image_encoder/
│   └── 1/
│       └── model.onnx
└── clip_text_encoder/
    └── 1/
        └── model.onnx
        ```