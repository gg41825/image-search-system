import os
from pathlib import Path

# ========== Load .env file ==========
try:
    from dotenv import load_dotenv
    
    # .env file location
    env_path = Path(__file__).parent.parent / '.env'
    
    if env_path.exists():
        load_dotenv(env_path)
        if int(os.getenv("DEV_MODE", "0")):
            print(f"[Config] Loaded .env from: {env_path}")
    else:
        print(f"[Config] .env not found at {env_path}")
        print(f"[Config] Using environment variables or default values")
        
except ImportError:
    print("[Config] python-dotenv not installed")
    print("[Config] Using environment variables only")
    print("[Config] Install with: pip install python-dotenv")

# ========== Development Settings ==========
DEV_MODE = int(os.getenv("DEV_MODE", "0"))  # 0: Production, 1: Development

# ========== MongoDB Configuration ==========
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "products")

# ========== Model Configuration ==========
MODEL_BERT = "bert-base-uncased"
MODEL_DINO = "dinov2_vitb14"
TRITON_MODEL_DINO = "facebook/dinov2-base"

# ========== File Paths ==========
IMG_UPLOAD_DIR = os.getenv("IMG_UPLOAD_DIR", "./data/uploads")
INDEX_PATH = os.getenv("INDEX_PATH", "./data/cache/aligned_index.ann")
ID_MAP_PATH = os.getenv("ID_MAP_PATH", "./data/cache/id_map.json")

# ========== Indexing Configuration ==========
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "-1")) # SAMPLE_SIZE: Number of products to index. -1 = all products, > 0 = specific number

#  ========== GEMINI ==========
GEMINI_API_KEY = "" # This is only for app/scripts/gen_image_info

# ========== Debug Output ==========
if DEV_MODE:
    print(f"[Config] Development Mode: ON")
    print(f"[Config] MONGO_URI: {MONGO_URI[:30]}...")
    print(f"[Config] MONGO_DB_NAME: {MONGO_DB_NAME}")
    print(f"[Config] SAMPLE_SIZE: {SAMPLE_SIZE}")