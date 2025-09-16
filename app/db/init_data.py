from typing import List, Dict, Tuple
import os
import json
import numpy as np
from db.mongo_client import MongoDBHandler
from models.bert_embedder import BERTEmbedder
from annoy import AnnoyIndex
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

def seed_product_vectors_bert(
    mongo: MongoDBHandler, sample_ids: List[str], num_trees: int = 10
) -> Tuple[AnnoyIndex, int, Dict[int, str]]:
    """
    Build Annoy vector index from product names + categories in MongoDB.

    Args:
        mongo (MongoDBHandler): MongoDB handler instance.
        num_trees (int): Number of trees to build in Annoy index.

    Returns:
        Tuple:
            - AnnoyIndex: The built Annoy index.
            - int: Embedding dimension.
            - Dict[int, str]: Mapping from Annoy idx -> MongoDB _id.
    """
    docs  = list(mongo.products.find(
        {"id": {"$in": sample_ids}},
        {"id": 1, "name": 1, "category": 1}
    )) if sample_ids else list(mongo.products.find({}, {"id": 1, "name": 1, "category": 1}))
    if not docs:
        raise RuntimeError("❌ No products found in MongoDB")

    texts = [f"{d['name']} {d['category']}" for d in docs]
    ids = [str(d["id"]) for d in docs]

    # Generate BERT embeddings
    bert = BERTEmbedder()
    vecs = bert.embed_texts(texts).cpu().numpy()
    dim = vecs.shape[1]

    # Build Annoy index
    index = AnnoyIndex(dim, "angular")
    id_map: Dict[int, str] = {}

    for i, (vec, pid) in enumerate(zip(vecs, ids)):
        index.add_item(i, vec.tolist())
        id_map[i] = pid

    index.build(num_trees)
    print(f"✅ Built Annoy index (BERT - text) with {len(ids)} products (dim={dim})")

    return index, dim, id_map

def seed_product_vectors_dino(mongo: MongoDBHandler, sample_ids: List[str], num_trees: int = 10):
    """
    Build Annoy vector index from product images in MongoDB using DINOv2.

    Args:
        mongo (MongoDBHandler): MongoDB handler.
        num_trees (int): Number of trees for Annoy index.

    Returns:
        index (AnnoyIndex): The built Annoy index.
        dim (int): Embedding dimension.
        id_map (dict[int, str]): Mapping from Annoy idx -> MongoDB _id.
    """
    # Fetch products
    docs  = list(mongo.products.find(
        {"id": {"$in": sample_ids}},
        {"id": 1, "image_url": 1}
    )) if len(sample_ids) > 0 else list(mongo.products.find({}, {"id": 1, "image_url": 1}))
    if not docs:
        raise RuntimeError("❌ No products found in MongoDB")

    ids = [str(d["id"]) for d in docs]
    urls = [d["image_url"] for d in docs]

    print(f"📦 Loaded {len(urls)} image URLs from MongoDB")

    # Initialize DINOv2
    dino = DINOv2Embedder(model_name="dinov2_vitb14")

    # Generate embeddings
    vecs = dino.embed_images(urls).cpu().numpy()
    dim = vecs.shape[1]

    # Build Annoy index
    index = AnnoyIndex(dim, "angular")
    id_map = {}

    for i, (vec, pid) in enumerate(zip(vecs, ids)):
        index.add_item(i, vec.tolist())
        id_map[i] = pid  # Annoy internal id -> Mongo _id

    index.build(num_trees)
    print(f"✅ Built Annoy index (Dino - image) with {len(ids)} images (dim={dim})")

    return index, dim, id_map

def seed_product_vectors_aligned(
    mongo: MongoDBHandler, sample_ids: List[str], num_trees: int = 10, alpha: float = 0.5
) -> Tuple[AnnoyIndex, int, Dict[int, str]]:
    """
    Build an Annoy index by aligning BERT (text) and DINO (image) embeddings.
    Uses simple concatenation of normalized vectors.

    Args:
        mongo (MongoDBHandler): MongoDB handler instance.
        sample_ids (List[str]): Product IDs to embed.
        num_trees (int): Number of trees to build in Annoy index.
        alpha (float): Weight between text and image embeddings (used if fusion instead of concat).

    Returns:
        Tuple:
            - AnnoyIndex: The built Annoy index.
            - int: Embedding dimension.
            - Dict[int, str]: Mapping from Annoy idx -> MongoDB _id.
    """
    # --- Fetch product text ---
    docs_text = list(mongo.products.find(
        {"id": {"$in": sample_ids}},
        {"id": 1, "name": 1, "category": 1}
    )) if sample_ids else list(mongo.products.find({}, {"id": 1, "name": 1, "category": 1}))
    if not docs_text:
        raise RuntimeError("❌ No products found for BERT embedding")

    texts = [f"{d['name']} {d['category']}" for d in docs_text]
    ids = [str(d["id"]) for d in docs_text]

    bert = BERTEmbedder()
    vecs_text = bert.embed_texts(texts).cpu().numpy()

    # --- Fetch product images ---
    docs_img  = list(mongo.products.find(
        {"id": {"$in": sample_ids}},
        {"id": 1, "image_url": 1}
    )) if len(sample_ids) > 0 else list(mongo.products.find({}, {"id": 1, "image_url": 1}))
    if not docs_img:
        raise RuntimeError("❌ No products found for DINO embedding")

    urls = [d["image_url"] for d in docs_img]
    dino = DINOv2Embedder(model_name="dinov2_vitb14")
    vecs_img = dino.embed_images(urls).cpu().numpy()

    # --- Align embeddings ---
    if vecs_text.shape[0] != vecs_img.shape[0]:
        raise RuntimeError("❌ Mismatch between text and image embeddings")

    # Normalize
    vecs_text = vecs_text / np.linalg.norm(vecs_text, axis=1, keepdims=True)
    vecs_img = vecs_img / np.linalg.norm(vecs_img, axis=1, keepdims=True)

    # Option 1: Concatenate
    combined_vecs = np.concatenate([vecs_text, vecs_img], axis=1)
    dim = combined_vecs.shape[1]
    
    # (Alternative Option: Weighted sum if same dim)
    # combined_vecs = alpha * vecs_text + (1 - alpha) * vecs_img
    # dim = combined_vecs.shape[1]

    # --- Build Annoy index ---
    index = AnnoyIndex(dim, "angular")
    id_map: Dict[int, str] = {}

    for i, (vec, pid) in enumerate(zip(combined_vecs, ids)):
        index.add_item(i, vec.tolist())
        id_map[i] = pid

    index.build(num_trees)
    print(f"✅ Built Annoy index (Aligned BERT+DINO) with {len(ids)} products (dim={dim})")

    return index, dim, id_map