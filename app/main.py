import os
import json
from typing import Optional
from fastapi import FastAPI, Query

import config
from pipeline.apis import run_search
from db.mongo_client import MongoDBHandler
from db.init_data import seed_products, seed_product_vectors_aligned

# -------------------------------------------------------------------
# Environment settings for stability (macOS M1/M2 + CPU)
# -------------------------------------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

app = FastAPI()

@app.on_event("startup")
def seed_db():
    # Initialize MongoDB connection using custom handler
    mongo = MongoDBHandler()

    # Seed products collection in MongoDB
    # If force_drop=True, it will drop existing products and reload from products.json
    seed_products(mongo, force_drop=True)

    # Fetch a list of product IDs (optionally sampled for testing/debugging)
    sample_ids = mongo.get_sample_ids(sample_size=config.SAMPLE_SIZE)

    # Build Annoy index by combining BERT (text) and DINO (image) embeddings
    # Returns: AnnoyIndex, embedding dimension, and id_map (Annoy internal ID -> product ID)
    index, dim, id_map = seed_product_vectors_aligned(mongo, sample_ids=sample_ids)

    # Save index
    # id_map: maps Annoy index ID to product ID in MongoDB
    # dim: embedding dimension, required to reload Annoy index
    index.save(config.INDEX_PATH)
    with open(config.ID_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump({"id_map": id_map, "dim": dim}, f)

    print(f"✅ Seeding finished.\nIndex saved at {config.INDEX_PATH}\n✅ id_map and dim saved at {config.ID_MAP_PATH} (dim={dim})")

@app.get("/search")
def search(
    query_text: Optional[str] = Query(None, description="Text query for product search"),
    image_url: str = Query(..., description="Image URL for product search"),
    embedder: str = Query("local", description="Choose between 'local' or 'triton'")
):
    results = run_search(query_text=query_text, query_image_url=image_url, embedder_type=embedder)
    return results
