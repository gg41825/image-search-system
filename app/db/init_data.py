from typing import List, Dict, Tuple
import hashlib
import os
import json
import time
import numpy as np

from annoy import AnnoyIndex
from db.mongo_client import MongoDBHandler
from models.bert_embedder import BERTEmbedder
from models.dino_embedder import DINOv2Embedder

def seed_products(mongo: "MongoDBHandler", force_drop: bool = False) -> List:
    """
    Seed the products collection with metadata from ../products.json.

    Args:
        mongo (MongoDBHandler): MongoDB handler instance.
        force_drop (bool): If True, drop existing collection and reseed.
                           If False, only insert missing products.

    Returns:
        List: Inserted product IDs (empty if nothing inserted).
    """
    if force_drop:
        mongo.products.drop()

    # Read metadata from products.json
    json_path = os.path.join(os.path.dirname(__file__), "../data/products.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"❌ products.json not found at {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        products = json.load(f)

    if not isinstance(products, list):
        raise ValueError("❌ products.json must contain a list of product dicts.")

    # Validate minimal required keys
    required_keys = {"id", "name", "category", "price", "image_url"}
    for i, product in enumerate(products):
        missing = required_keys - product.keys()
        if missing:
            raise ValueError(f"❌ Product at index {i} is missing required fields: {missing}")

    # Get already existing product ids
    existing_ids = set(
        doc["id"] for doc in mongo.products.find({}, {"id": 1})
    )

    # Filter only new products
    new_products = [p for p in products if p["id"] not in existing_ids]

    inserted_ids = []
    if new_products:
        result = mongo.products.insert_many(new_products)
        inserted_ids = result.inserted_ids

    print(f"✅ Inserted {len(inserted_ids)} new products into MongoDB")

    return inserted_ids


def seed_product_vectors_aligned(
    mongo: MongoDBHandler,
    sample_ids: List[str],
    num_trees: int = 5,   # smaller = faster build
    alpha: float = 0.5,
    batch_size: int = 32,
    cache_dir: str = "data/cache",
    cache_expiry_days: int = 30
) -> Tuple[AnnoyIndex, int, Dict[int, str]]:
    """
    Build an Annoy index by aligning BERT (text) and DINO (image) embeddings.
    Supports caching and batch embedding to speed up processing.
    Cache filename depends on sample_ids + expires after N days.
    """

    os.makedirs(cache_dir, exist_ok=True)

    # --------------------------
    # Fetch product metadata
    # --------------------------
    docs_text = list(mongo.products.find(
        {"id": {"$in": sample_ids}},
        {"id": 1, "name": 1, "category": 1}
    )) if sample_ids else list(mongo.products.find({}, {"id": 1, "name": 1, "category": 1}))
    if not docs_text:
        raise RuntimeError("❌ No products found for BERT embedding")

    texts = [f"{d['name']} {d['category']}" for d in docs_text]
    ids = [str(d["id"]) for d in docs_text]

    docs_img = list(mongo.products.find(
        {"id": {"$in": sample_ids}},
        {"id": 1, "image_url": 1}
    )) if sample_ids else list(mongo.products.find({}, {"id": 1, "image_url": 1}))
    if not docs_img:
        raise RuntimeError("❌ No products found for DINO embedding")

    urls = [d["image_url"] for d in docs_img]

    # --------------------------
    # Generate cache key
    # --------------------------
    key = ",".join(sorted(sample_ids)) if sample_ids else "all"
    key_hash = hashlib.md5(key.encode()).hexdigest()[:8]

    text_cache = os.path.join(cache_dir, f"text_emb_{key_hash}.npy")
    img_cache = os.path.join(cache_dir, f"img_emb_{key_hash}.npy")

    def is_cache_valid(path: str) -> bool:
        """Check if cache exists and not expired."""
        if not os.path.exists(path):
            return False
        file_age_days = (time.time() - os.path.getmtime(path)) / (3600 * 24)
        return file_age_days <= cache_expiry_days

    # --------------------------
    # Load from cache or rebuild
    # --------------------------
    if is_cache_valid(text_cache) and is_cache_valid(img_cache):
        print(f"⚡ Loading embeddings from cache ({key_hash})...")
        vecs_text = np.load(text_cache)
        vecs_img = np.load(img_cache)
    else:
        print(f"⚡ Rebuilding cache for {key_hash} (expired or missing)...")

        # --- Encode text in batches ---
        bert = BERTEmbedder()
        text_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            vecs = bert.embed_texts(batch).cpu().numpy()
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
            text_embs.append(vecs)
        vecs_text = np.vstack(text_embs)

        # --- Encode images in batches ---
        dino = DINOv2Embedder(model_name="dinov2_vitb14")
        img_embs = []
        for i in range(0, len(urls), batch_size):
            batch = urls[i:i+batch_size]
            vecs = dino.embed_images(batch).cpu().numpy()
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
            img_embs.append(vecs)
        vecs_img = np.vstack(img_embs)

        # Save cache
        np.save(text_cache, vecs_text)
        np.save(img_cache, vecs_img)

    # --------------------------
    # Align embeddings (concat)
    # --------------------------
    if vecs_text.shape[0] != vecs_img.shape[0]:
        raise RuntimeError("❌ Mismatch between text and image embeddings")

    combined_vecs = np.concatenate([vecs_text, vecs_img], axis=1)
    dim = combined_vecs.shape[1]

    # --------------------------
    # Build Annoy index
    # --------------------------
    index = AnnoyIndex(dim, "angular")
    id_map: Dict[int, str] = {}

    for i, (vec, pid) in enumerate(zip(combined_vecs, ids)):
        index.add_item(i, vec.tolist())
        id_map[i] = pid

    index.build(num_trees)
    print(f"✅ Built Annoy index (Aligned BERT+DINO) with {len(ids)} products (dim={dim}, cache={key_hash})")

    return index, dim, id_map