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

> **⏳ Note on Cold Start**: The first request may take 1-2 minutes as the Cloud Run instance starts up and loads the ML models. Subsequent requests will be much faster (1-5 seconds).

![Demo GIF showing the search process](demo.gif)

---

# ⚙️ How it Works

### 1. Data

Products (name, category, image) are stored in MongoDB Atlas.

Example:
```json
{
  "name": "Casio Quartz Analog Watch",
  "category": "Watches",
  "image_file": "0.jpg",
  "image_url": "https://pub-6cf2f88db8f14219bf79c4d284c2c63e.r2.dev/0.jpg",
  "id": 0
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
curl -X POST "https://image-search-system-495449323600.europe-west1.run.app/search" \
  -F "image_url=https://pub-xxx.r2.dev/1.jpg" \
  -F "query_text=dark blue jacket"
```

Example response:
```json
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

# 🏗️ Architecture
```
┌─────────────────┐
│   User Request  │
└────────┬────────┘
         │
         v
┌────────────────────────────┐
│   Cloud Run Service        │
│  ┌─────────────────────┐   │
│  │  FastAPI Backend    │   │
│  │  + Frontend (HTML)  │   │
│  └──────────┬──────────┘   │
│             │              │
│  ┌──────────v──────────┐   │
│  │  TritonEmbedder     │   │
│  │  (BERT + DINOv2)    │   │
│  └──────────┬──────────┘   │
│             │              │
│  ┌──────────v──────────┐   │
│  │  Annoy Index        │   │
│  │  (Vector Search)    │   │
│  └──────────┬──────────┘   │
└─────────────┼──────────────┘
              │
              v
     ┌────────────────┐
     │ MongoDB Atlas  │
     │ (Metadata DB)  │
     └────────────────┘
```

### Key Components

- **Cloud Run**: Serverless container platform (scales to zero when idle)
- **FastAPI**: Python web framework for API and static file serving
- **PyTorch**: Direct model inference (BERT + DINOv2)
- **Annoy**: Fast approximate nearest neighbor search
- **MongoDB Atlas**: Cloud-hosted NoSQL database for product metadata
- **Cloudflare R2**: Object storage for product images

---

# 🛠️ Development Notes

### Configuration

Configuration is set in `app/config.py` and can be overridden by environment variables:

**GitHub Actions Secrets** (set in repository settings):
- `GCP_PROJECT_ID` → Google Cloud project ID
- `GCP_SA_KEY` → Google Cloud service account key (base64 encoded)
- `HF_TOKEN` → Hugging Face token for model access
- `MONGO_URI` → MongoDB Atlas connection string

**Cloud Run Environment Variables** (defined in `service.template.yaml`):
- `DEV_MODE` → Enable development mode (default: `0`)
- `IMG_UPLOAD_DIR` → Temporary directory for uploaded images (default: `/tmp/uploads`)
- `INDEX_PATH` → Path to store Annoy index (default: `/app/data/cache/aligned_index.ann`)
- `ID_MAP_PATH` → Path to store ID mapping (default: `/app/data/cache/id_map.json`)

**GitHub Actions Workflow Variables** (defined in `.github/workflows/deploy.yml`):
- `SAMPLE_SIZE` → Number of products to index (default: `20`)
  - Can be modified in the workflow file or via manual workflow dispatch
  - Larger values increase startup time but provide more search results

### Local Development

#### 1. Set up Python virtual environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r app/requirements.txt
```

#### 2. Set environment variables
```bash
cp .env.example .env
```
Then edit .env with your values

#### 3. Run the application
```bash
cd app
uvicorn main:app --reload --host 0.0.0.0 --port 8080
```

Visit: http://localhost:8080

#### 4. Deactivate virtual environment (when done)
```bash
deactivate
```

### Deployment

**Automatic Deployment** (via GitHub Actions):

Push to `cloud-run-deployment` branch to trigger automatic deployment:
```bash
git push origin main:cloud-run-deployment
```

**Manual Deployment** (via gcloud CLI):
```bash
# Build and deploy
gcloud builds submit --config=cloudbuild.yaml
gcloud run services replace service.template.yaml --region=europe-west1
```

### API Documentation

- **Swagger UI**: [https://image-search-system-495449323600.europe-west1.run.app/docs](https://image-search-system-495449323600.europe-west1.run.app/docs)
- **ReDoc**: [https://image-search-system-495449323600.europe-west1.run.app/redoc](https://image-search-system-495449323600.europe-west1.run.app/redoc)

---

# 🧪 Testing

### Via Web Interface
1. Visit [https://image-search-system-495449323600.europe-west1.run.app/](https://image-search-system-495449323600.europe-west1.run.app/)
2. Upload an image or paste an image URL
3. Optionally add a text query
4. Click "Search"

### Via API (curl)
```bash
# Search by image URL
curl -X POST "https://image-search-system-495449323600.europe-west1.run.app/search" \
  -F "image_url=https://pub-6cf2f88db8f14219bf79c4d284c2c63e.r2.dev/0.jpg"

# Search by image URL + text
curl -X POST "https://image-search-system-495449323600.europe-west1.run.app/search" \
  -F "image_url=https://pub-6cf2f88db8f14219bf79c4d284c2c63e.r2.dev/0.jpg" \
  -F "query_text=watch"

# Upload local image
curl -X POST "https://image-search-system-495449323600.europe-west1.run.app/search" \
  -F "file=@/path/to/image.jpg" \
  -F "query_text=blue jacket"
```

---

# 📊 Performance

- **Cold Start**: ~60-120 seconds (first request after idle period)
- **Warm Request**: ~1-5 seconds (when instance is already running)
- **Index Build Time**: ~2-3 minutes for 20 products (scales linearly)
- **Search Latency**: ~500ms-2s per query

**Cost Optimization**:
- Cloud Run scales to zero when idle (no requests = no cost)
- Estimated monthly cost: €0-5 for demo/portfolio use
- MongoDB Atlas M0 (free tier): 512MB storage

---

# 🔧 Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Backend** | FastAPI (Python) | API server and static file serving |
| **ML Models** | BERT + DINOv2 (PyTorch) | Text and image embeddings |
| **Vector Search** | Annoy | Fast approximate nearest neighbor search |
| **Database** | MongoDB Atlas | Product metadata storage |
| **Image Storage** | Cloudflare R2 | Product image hosting |
| **Deployment** | Google Cloud Run | Serverless container platform |
| **CI/CD** | GitHub Actions | Automated build and deployment |
| **Frontend** | HTML + CSS + Vanilla JS | Simple web interface |

---

# 📁 Project Structure
```
.
├── .github/
│   └── workflows/
│       └── deploy.yml              # GitHub Actions CI/CD
├── app/
│   ├── main.py                     # FastAPI application entry point
│   ├── config.py                   # Configuration management
│   ├── requirements.txt            # Python dependencies
│   ├── db/
│   │   ├── mongo_client.py        # MongoDB connection handler
│   │   └── init_data.py           # Data seeding and index building
│   ├── models/
│   │   └── triton_embedder.py     # BERT + DINOv2 embedder
│   ├── pipeline/
│   │   └── search.py              # Search logic
│   ├── routers/
│   │   └── search_router.py       # API endpoints
│   └── frontend/
│       ├── index.html             # Web interface
│       ├── style.css              # Styling
│       └── script.js              # Frontend logic
├── Dockerfile                      # Container definition
├── service.template.yaml           # Cloud Run service configuration
└── README.md                       # This file
```

---

# 🚨 Known Limitations

- **Cold Start Latency**: First request after ~15 minutes of inactivity takes 1-2 minutes
- **Limited Dataset**: Only 194 products indexed (for demo purposes)
- **Single Instance**: No horizontal scaling configured (max 3 instances)
- **No Caching**: Each search performs full vector computation
- **No Authentication**: Public API (suitable for demo only)

---

# 📝 Disclaimer

This project is for personal or educational use only.

- All product images and data belong to their respective owners
- Only 194 images and corresponding metadata from the open-source [GLAMI-1M dataset](https://github.com/glami/glami-1m) are used
- Images are hosted on self-hosted Cloudflare R2
- 5 test images are used solely for testing purposes
- This is a demonstration project and not intended for production use

---

# 📄 License

MIT License - see [LICENSE](LICENSE) file for details

---

# 🙏 Acknowledgments

- [GLAMI-1M Dataset](https://github.com/glami/glami-1m) for product data
- [Spotify Annoy](https://github.com/spotify/annoy) for fast vector search
- [Hugging Face](https://huggingface.co/) for pre-trained models (BERT, DINOv2)
- [FastAPI](https://fastapi.tiangolo.com/) for the excellent web framework
- Google Cloud Platform for serverless infrastructure