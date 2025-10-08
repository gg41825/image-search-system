# 🧭 Project Overview

This project demonstrates a **multimodal product search system**:  
- You provide either an **image** (mandatory) and/or a **text query** (optional).  
- The system performs a **nearest neighbor search** in a **vector database** built with [Annoy](https://github.com/spotify/annoy).  
- It combines **BERT** (text embeddings) + **DINOv2** (image embeddings), aligned via **ONNX/Triton Inference Server** for fast inference.  
- The best-matching product metadata (from **MongoDB**) is returned.  

👉 **Use case**:  
Imagine shopping online. You upload a photo of your hiking jacket and type *“dark blue jacket”* → the system retrieves the most relevant product from the catalog.

---
# 🐍 Local Development (Optional)
If you prefer to run the project locally without Docker:
## 1. If you prefer to run the project locally without Docker:
```bash
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

## 2. Install dependencies
```bash
pip install -r requirements.txt
```

## 3. Make entrypoint.sh executable (first time only)
```bash
chmod +x ./entrypoint.sh
```

## 4. Export the ONNX model (via entrypoint.sh)
```bash
./entrypoint.sh true
```

It will generate model.onnx under `model_repository/aligned/1`:
```
model_repository/
  aligned/
    1/
      model.onnx
    config.pbtxt
```

## 4. Start MongoDB and Triton manually with Docker

```bash
docker run -d --name mongodb \
  -p 27017:27017 \
  -v ~/mongo_data:/data/db \
  -e MONGO_INITDB_ROOT_USERNAME=root \
  -e MONGO_INITDB_ROOT_PASSWORD=iamtestginny \
  mongo:7.0
```

```bash
docker run --rm -it -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "$(pwd)/model_repository:/models" \
  nvcr.io/nvidia/tritonserver:23.10-py3 \
  tritonserver --model-repository=/models --disable-auto-complete-config
```

## 5. Run the app
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

---

# 🚀 Deployment with Docker Compose

We provide a `docker-compose.yml` that runs the whole pipeline:

- **MongoDB** → Stores product metadata and vectors.  
- **Triton Inference Server** → Runs the ONNX model for fast aligned embedding.  
- **App (FastAPI)** → Handles API requests and frontend.  

## 1. Clone the repository
```bash
git clone https://github.com/<your-repo>.git
cd <your-repo>
```

## 2. Build and start all services
```bash
docker compose up --build
```

This will start:

- mongodb → on port 27017

- triton → on ports 8000, 8001, 8002

- app (FastAPI) → on port 8080

## 3. Access the system

API Docs (Swagger UI): 👉 http://localhost:8080/docs

Frontend (demo UI): 👉 http://localhost:8080/

# ⚙️ How it Works
### 1. Data

Products (brand, title, category, price, image) are stored in MongoDB.

Example:
```bash
{
  "id": "1",
  "name": "Patagonia - Worn Wear Patch Kit Reparaturset",
  "category": "Accessoires",
  "price": 24.95,
  "image_url": "https://pub-xxx.r2.dev/1.jpg",
  "original_image_url": "https://www.bfgcdn.com/out/pictures/...jpg"
}
```
### 2. Indexing

- At app startup, products are **seeded** into MongoDB.

- `Annoy` builds an **approximate nearest neighbor (ANN)** index:

  - **BERT** → encodes text (name + category).

  - **DINOv2** → encodes images.

  - Both embeddings are normalized and concatenated → stored in index.

### 3. Query

API `/search` accepts:

- Uploaded image file or image URL (required)

- Text query (optional)

Example request:
```bash
curl -X POST "http://localhost:8080/search" \
  -F "image_url=https://pub-xxx.r2.dev/1.jpg" \
  -F "query_text=dark blue jacket"
```
Example response:
```bash
{
  "results": [
    {
      "id": "123",
      "name": "CMP Jacket",
      "category": "Outdoor",
      "price": 23.90,
      "image_url": "https://...",
      "distance": 0.4231
    }
  ]
}
```

# 🛠️ Development Notes

- Configuration is set in app/config.py and can be overridden by Docker env variables:
  - `DEV_MODE`

  - `MONGO_URI`

  - `TRITON_URL`

  - `INDEX_PATH`

  - `ID_MAP_PATH`

  - `SAMPLE_SIZE`

- **Frontend** is served via FastAPI static files (`/frontend`).

- **Swagger UI** makes testing APIs easier.