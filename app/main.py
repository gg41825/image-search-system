import argparse
import sys
import os
import numpy as np
from db.mongo_client import MongoDBHandler
from db.init_data import seed_products, seed_product_vectors_aligned
import config

from models.local_embedder import LocalEmbedder
from models.triton_embedder import TritonEmbedder

# -------------------------------------------------------------------
# Environment settings for stability (macOS M1/M2 + CPU)
# -------------------------------------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def run_with_embedder(query_text: str, query_image_url: str, embedder_type: str):
    # Init Mongo DB
    mongo = MongoDBHandler()
    seed_products(mongo, force_drop=False)

    sample_ids = mongo.get_sample_ids(config.SAMPLE_SIZE) if config.SAMPLE_SIZE > 0 else []
    docs = list(mongo.products.find({}, {"id": 1, "name": 1, "category": 1}))
    if not docs:
        print("❌ Products collection is empty even after seeding. Exiting.")
        sys.exit(1)

    index, dim, id_map = seed_product_vectors_aligned(mongo, sample_ids=sample_ids)

    # ---- Choose embedder ----
    if not query_text and not query_image_url:
        print("❌ You must provide at least --query_text or --image_url")
        sys.exit(1)

    if embedder_type == "local":
        print("⚡ Using Local Embedder")
        embedder = LocalEmbedder()
        query_combined = embedder.embed(
            texts=[query_text] if query_text else None,
            images=[query_image_url] if query_image_url else None
        )

    elif embedder_type == "triton":
        print("⚡ Using Triton Embedder (remote inference)")
        triton = TritonEmbedder(url=config.TRITON_URL, model_name="aligned")
        query_combined = triton.embed(query_text, query_image_url)

    else:
        raise ValueError("embedder_type must be 'local' or 'triton'")

    # ---- Search ----
    top_k = 1 # Only find the best match
    nn_indices, distances = index.get_nns_by_vector(
        query_combined[0].tolist(), top_k, include_distances=True
    )
    pids = [str(id_map[idx]) for idx in nn_indices]

    docs = list(mongo.products.find(
        {"id": {"$in": pids}},
        {"_id": 0, "id": 1, "name": 1, "category": 1, "image_url": 1}
    ))
    doc_map = {doc["id"]: doc for doc in docs}

    print("🔍 Query results:")
    for idx, dist in zip(nn_indices, distances):
        pid = str(id_map[idx])
        doc = doc_map.get(pid)
        if doc:
            print(f" - ID: {doc['id']}, "
                  f"Name: {doc.get('name')}, "
                  f"Category: {doc.get('category')}, "
                  f"Image: {doc.get('image_url')}, "
                  f"Distance={dist:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Product Search with Text + Image query")
    parser.add_argument("--query_text", type=str, default="", help="Text query for product search")
    parser.add_argument("--image_url", type=str, required=True, help="Image URL for product search")
    parser.add_argument("--embedder", type=str, default="triton", choices=["local", "triton"],
                        help="Choose which embedder to use: local (BERT+DINO) or triton (remote inference)")

    args = parser.parse_args()
    run_with_embedder(query_text=args.query_text, query_image_url=args.image_url, embedder_type=args.embedder)
