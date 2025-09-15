from typing import List, Dict, Tuple
import os
import json
from db.mongo_client import MongoDBHandler
from models.bert_embedder import BERTEmbedder
from annoy import AnnoyIndex


def seed_products(mongo: MongoDBHandler, force_drop: bool = False) -> List:
    """
    Seed the products collection with metadata from ../products.json.

    Args:
        mongo (MongoDBHandler): MongoDB handler instance.
        force_drop (bool): If True, drop existing collection and reseed.
                           If False, only seed if collection is empty.

    Returns:
        List: Inserted product IDs (empty if nothing inserted).
    """
    existing_count = mongo.products.count_documents({})

    # Skip seeding if documents already exist (unless force_drop is True)
    if existing_count > 0 and not force_drop:
        print(f"⚠️ Products collection already has {existing_count} documents. Skipping seeding.")
        return []

    if force_drop:
        mongo.products.drop()

    # Read metadata from products.json
    json_path = os.path.join(os.path.dirname(__file__), "../data/products.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"❌ products.json not found at {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if not isinstance(metadata, list):
        raise ValueError("❌ products.json must contain a list of product dicts.")

    # Optional: validate minimal required keys for each product
    required_keys = {"_id", "name", "category", "price", "image_url"}
    for i, item in enumerate(metadata):
        if not required_keys.issubset(item.keys()):
            raise ValueError(f"❌ Product at index {i} is missing required fields: {required_keys - item.keys()}")

    # Insert products into MongoDB
    inserted_ids = mongo.insert_products(metadata)
    print(f"✅ Inserted {len(inserted_ids)} products into MongoDB")

    return inserted_ids

def seed_product_vectors_bert(
    mongo: MongoDBHandler, num_trees: int = 10
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
    docs = list(mongo.products.find({}, {"_id": 1, "name": 1, "category": 1}))
    if not docs:
        raise RuntimeError("❌ No products found in MongoDB")

    texts = [f"{d['name']} {d['category']}" for d in docs]
    ids = [str(d["_id"]) for d in docs]

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
    print(f"✅ Built Annoy index with {len(ids)} products (dim={dim})")

    return index, dim, id_map