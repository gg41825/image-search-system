# 🧭 Project Overview

This project demonstrates a **multimodal product search system**:  
- You provide either an **image** (mandatory) and/or a **text query** (optional).  
- The system performs a **nearest neighbor search** in a **vector database** built with [Annoy](https://github.com/spotify/annoy).  
- It combines **BERT** (text embeddings) + **DINOv2** (image embeddings), aligned via **ONNX/Triton Inference Server** for fast inference.  
- The best-matching product metadata (from **MongoDB**) is returned.  

👉 **Use case**:  
Imagine shopping online. You upload a photo of your hiking jacket and type *“dark blue jacket”* → the system retrieves the most relevant product from the catalog.

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
docker-compose up --build
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