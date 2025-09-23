import os

MONGO_URI = os.getenv("MONGO_URI", "mongodb://root:iamtestginny@localhost:27017")
MONGO_DB_NAME = "product_db"

MODEL_BERT = "bert-base-uncased"
MODEL_DINO = "dinov2_vitb14"
TRIOTON_MODEL_DINO = "facebook/dinov2-base"

# Temporary upload dir
IMG_UPLOAD_DIR = os.getenv("IMG_UPLOAD_DIR", "/app/uploads")

# SAMPLE_SIZE must be greater than 0
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "50"))

# Triton Inference Server REST endpoint
TRITON_URL = os.getenv("TRITON_URL", "http://localhost:8000")
TRITON_MODEL_NAME = "aligned"

# Path for Annoy index and id_map
INDEX_PATH = os.getenv("INDEX_PATH", "/app/data/aligned_index.ann")
ID_MAP_PATH = os.getenv("ID_MAP_PATH", "/app/data/id_map.json")