import argparse
import sys
import os
import numpy as np
from db.mongo_client import MongoDBHandler
from db.init_data import seed_products, seed_product_vectors_aligned
import config

from models.bert_embedder import BERTEmbedder
from models.dino_embedder import DINOv2Embedder
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

    # Seed products
    try:
        seed_products(mongo, force_drop=False)
    except Exception as e:
        mongo.log_error(
            error_type="data",
            message=str(e),
            stacktrace="Error occurred when seed_products."
        )
        sys.exit(1)

    sample_ids = mongo.get_sample_ids(config.SAMPLE_SIZE) if config.SAMPLE_SIZE > 0 else []
    docs = list(mongo.products.find({}, {"id": 1, "name": 1, "category": 1}))
    if not docs:
        print("❌ Products collection is empty even after seeding. Exiting.")
        sys.exit(1)

    try:
        index, dim, id_map = seed_product_vectors_aligned(mongo, sample_ids=sample_ids)
    except Exception as e:
        print(f"❌ Failed to build vector DB: {e}")
        sys.exit(1)

    # ---- Choose embedder ----
    if embedder_type == "local":
        print("⚡ Using Local Embedder (BERT + DINOv2)")
        bert = BERTEmbedder()
        dino = DINOv2Embedder()
        query_vec = bert.embed_texts([query_text]).cpu().numpy()
        query_img_vec = dino.embed_images([query_image_url]).cpu().numpy()
    elif embedder_type == "triton":
        print("⚡ Using Triton Embedder (remote inference)")
        triton = TritonEmbedder(url=config.TRITON_URL, model_name=config.TRITON_MODEL_NAME)
        query_vec = triton.embed(query_text, query_image_url)
        query_img_vec = np.zeros_like(query_vec)  # already aligned → no need extra
    else:
        raise ValueError("embedder_type must be 'local' or 'triton'")

    # ---- Normalize + concat (if local) ----
    if embedder_type == "local":
        query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
        query_img_vec = query_img_vec / np.linalg.norm(query_img_vec, axis=1, keepdims=True)
        query_combined = np.concatenate([query_vec, query_img_vec], axis=1)
    else:
        query_combined = query_vec  # Triton already outputs aligned vector

    # ---- Search ----
    top_k = 1
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
    parser.add_argument("--query_text", type=str, required=True, help="Text query for product search")
    parser.add_argument("--image_url", type=str, required=True, help="Image URL for product search")
    parser.add_argument("--embedder", type=str, default="triton", choices=["local", "triton"],
                        help="Choose which embedder to use: local (BERT+DINO) or triton (remote inference)")

    args = parser.parse_args()
    run_with_embedder(query_text=args.query_text, query_image_url=args.image_url, embedder_type=args.embedder)
