# Project Overview

Provide one picture (or with text), and the system will perform a nearest neighbor search in the vector database. Then return the best match from MongoDB.

# 1. Environment Setup

Make sure to have Python 3.10+ and pip installed.
I am developing this on a MacBook Pro with an M2 Max chip, so some of the configs lean Apple Silicon–friendly.

## 1-1. Create a virtual environment (recommended)
```
python3 -m venv venv
source venv/bin/activate # Mac
venv\Scripts\activate # Command Prompt
```

## 1-2. Install dependencies
```
pip install -r requirements.txt
```

## 1-3. Run MongoDB with Docker

Start a MongoDB instance locally:
```
docker pull mongo:7.0
```
```
docker run -d --name mongodb \
  -p 27017:27017 \
  -v ~/mongo_data:/data/db \
  -e MONGO_INITDB_ROOT_USERNAME=root \
  -e MONGO_INITDB_ROOT_PASSWORD=iamtestginny \
  mongo:7.0
```
This will start MongoDB on localhost:27017.

## 1-4. Run the Triton CPU Docker image
Use the following command to pull and start Triton and expose the HTTP, gRPC, and metrics ports:

```
docker pull nvcr.io/nvidia/tritonserver:23.10-py3
```
```
docker run --rm -it -p 8000:8000 -p 8001:8001 -p 8002:8002 -v "$(pwd)/model_repository:/models" nvcr.io/nvidia/tritonserver:23.10-py3 tritonserver --model-repository=/models --disable-auto-complete-config
```

## 1-5. Export Bert+Dino Model to a Combined Model (Aligned)
```
PYTHONPATH=app python3 app/scripts/02_export_align_to_onnx.py
```
Then moved the generated model.onnx under `model_repository/aligned/1`

The structure should look like the following:
```
app/
model_repository/
  aligned/
    1/
      model.onnx
    config.pbtxt
```

# 2. Run the Pipeline

Run the main program with a text query and an image URL:
- At least one input (--query_text or --image_url) must be provided.
- Since I was testing on local, so I keep the `local` inference as well.


## 2-1. Correct Usage:
```
python3 app/main.py --image_url 'https://pub-6cf2f88db8f14219bf79c4d284c2c63e.r2.dev/1.jpg' --embedder local
```
```
python3 app/main.py --query_text "dark blue jacket" --image_url "https://pub-6cf2f88db8f14219bf79c4d284c2c63e.r2.dev/1.jpg" --embedder local
```
```
python3 app/main.py --image_url "https://pub-6cf2f88db8f14219bf79c4d284c2c63e.r2.dev/1.jpg" --embedder triton
```
```
python3 app/main.py --query_text "dark blue jacket" --image_url "https://pub-6cf2f88db8f14219bf79c4d284c2c63e.r2.dev/1.jpg" --embedder triton
```
#### args:
`--query_text`: The text query for product search. Default: "" (empty string)

`--image_url`: The image URL for product search. Required: Yes.

`--embedder`: Choices:

"local": Run everything locally using BERT + DINOv2 (slower but no server dependency).

"triton": Use Triton Inference Server (faster, aligned embedding).

## 2-2. Output
```
🔍 Query results:
 - ID: 123, Name: CMP Jacket, Category: Outdoor, Image: https://... , Distance=0.4231
 ```


# 3. Additional Note

## 3-1. Config File
`app/config.py` states the configuration of the program. If SAMPLE_SIZE > 0, only a subset of products will be indexed into Annoy (useful for debugging).

---

## 3-2.How I generate the Mongo DB data
I was browsing Bergfreunde.de for my hiking clothes when I received the task, so it kind of inspired me to use the data from the webste.

---

### 3-2-1. Implementation of Data Generation
Code is under `app/scripts/01_get_image_data.py`

- Crawls multiple product categories (defined in `categories_map`)
- Extracts:
  - Brand name
  - Product title
  - Category
  - Image URL
  - Price (converted to float, e.g., `"€ 54,97"` → `54.97`)
-  Combine Brand name, Product title into the field `name`
-  Downloads product images locally
- Uploads images to a my Cloudflare R2 public bucket, add the `image_url` for the corresponding image link. `original_image_url` is the original image url.

- Generate the complete product dataset in `scripts/products.json`. Remember to put it into `app/data/products.json`

### 3-2-2. Example entry in products.json:

```
{
  "id": "1",
  "name": "Patagonia - Worn Wear Patch Kit Reparaturset",
  "category": "Accessoires",
  "price": 24.95,
  "image_url": "https://pub-6cf2f88db8f14219bf79c4d284c2c63e.r2.dev/1.jpg",
  "original_image_url": "https://www.bfgcdn.com/out/pictures/generated/product/1/215_215_90/sol_005-5610-0311_pic1_1.jpg"
}
```

Please note that web scraping may violate a website’s terms of service. Therefore, all downloaded images will be deleted after 30 days.