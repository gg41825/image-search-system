# 🧭 Project Overview
This project demonstrates a **multimodal product search system**:  
- You provide either an image (mandatory, can be uploaded from local device or via URL) and/or a text query (optional). 
- The system performs a **nearest neighbor search** in a **vector database** built with [Annoy](https://github.com/spotify/annoy).  
- It combines BERT (text embeddings) + DINOv2 (image embeddings), both running directly in PyTorch for fast inference. (The source images are from [GLAMI-1M](https://github.com/glami/glami-1m))
- The best-matching product metadata (from MongoDB Atlas) is returned.

---

# 🚀 Live Deployment
The entire pipeline is deployed as a single Cloud Run service:

- API + frontend + model inference run in one container
- MongoDB uses a free online cluster (MongoDB Atlas)
- Fully serverless and scalable

You can try the live demo here:
[Live Demo on Cloud Run](https://image-search-system-495449323600.europe-west1.run.app/)
![Demo GIF showing the search process](demo.gif)

---

# ⚙️ How it Works
### 1. Data

Products (name, category, image) are stored in MongoDB Atlas.

Example:
```bash
{
  "name": "Casio Quartz Analog Watch",
  "category": "Watches",
  "image_file": "0.jpg",
  "image_url": "https://pub-6cf2f88db8f14219bf79c4d284c2c63e.r2.dev/0.jpg",
  "id": 0
},
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
curl -X POST "https://<your-cloud-run-url>/search" \
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
      "image_url": "https://...",
      "similarity": 0.4231
    }
  ]
}
```

---

# 🛠️ Development Notes

- Configuration is set in app/config.py and can be overridden by Docker env variables:
  Defined in GitHub Actions secrets:

  - `GCP_PROJECT_ID` → Google Cloud project ID

  - `GCP_SA_KEY` → Google Cloud service account key

  - `HF_TOKEN` → Hugging Face token for model access

  - `MONGO_URI` → MongoDB Atlas connection string
  
  Defined in Cloud Run:
  - `DEV_MODE` → Enable development mode

  - `IMG_UPLOAD_DIR` → Temporary directory for uploaded images

  - `INDEX_PATH` → Path to store Annoy index

  - `ID_MAP_PATH` → Path to store ID mapping

  Defined in GitHub Actions workflow:
  - `SAMPLE_SIZE` → Number of products to index. Default is 20. Can be modified in deploy.yml (https://github.com/YOUR_USERNAME/YOUR_REPO/actions/workflows/deploy.yml)

- **Frontend** is served via FastAPI static files (`/frontend`).

- [**Swagger UI**](https://image-search-system-495449323600.europe-west1.run.app/docs) makes testing APIs easier.

---

   # Disclaimer
   
   This project is for personal or educational use only.
All product images and data belong to their respective owners.
Only 194 images and the corresponding metadata from the open source data GLAMI-1M-dataset are stored in self-hosted Cloudflare R2, and 5 test images are only used for this project's test purpose.