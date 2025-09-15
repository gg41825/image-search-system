from db.mongo_client import MongoDBHandler
from db.init_data import seed_products, seed_product_vectors_bert, seed_product_vectors_dino, seed_product_vectors_aligned
import sys
import config
from models.bert_embedder import BERTEmbedder
from models.dino_embedder import DINOv2Embedder
import numpy as np

# -------------------------------------------------------------------
# Environment settings to improve stability on macOS (M1/M2) + CPU:
# - TOKENIZERS_PARALLELISM: disable multi-threaded Hugging Face tokenizer,
#   prevents extra worker processes and semaphore leaks.
# - OMP_NUM_THREADS / MKL_NUM_THREADS: limit thread count for PyTorch
#   and BLAS/LAPACK backends, avoids oversubscription and resource tracker warnings.
# -------------------------------------------------------------------
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def main():
    # Init Mongo DB
    mongo = MongoDBHandler()

    # Seed products to MongoDB
    try:
        seed_products(mongo, force_drop=False)
    except Exception as e:
        mongo.log_error(
            error_type="data",
            message=e,
            stacktrace="Error occured when seed_products."
        )
        sys.exit(1)  # stop program safely
    
    sample_ids = mongo.get_sample_ids(config.SAMPLE_SIZE) if config.SAMPLE_SIZE > 0 else []

    docs = list(mongo.products.find({}, {"id": 1, "name": 1, "category": 1}))
    if not docs:
        mongo.log_error(
            error_type="data",
            message="Products still missing after seed_products",
            stacktrace="MongoDB products collection is empty"
        )
        print("❌ Products collection is empty even after seeding. Exiting.")
        sys.exit(1)

    try:
        index, dim, id_map = seed_product_vectors_aligned(mongo, sample_ids)
    except Exception as e:
        mongo.log_error(
            error_type="vector_db",
            message="Failed to seed product images+text vectors",
            stacktrace=str(e)
        )
        print(f"❌ Failed to build vector DB: {e}")
        sys.exit(1)

    # ---- Query example ----
    query_text = "dark blue jacket"
    bert = BERTEmbedder()
    query_vec = bert.embed_texts([query_text]).cpu().numpy()

    # Example: if alignment was concatenation, also get image vec (dummy if no image available)
    dummy_image_url = "https://pub-6cf2f88db8f14219bf79c4d284c2c63e.r2.dev/1.jpg"
    dino = DINOv2Embedder()
    query_img_vec = dino.embed_images([dummy_image_url]).cpu().numpy()

    # Align query vectors the same way as the index
    query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
    query_img_vec = query_img_vec / np.linalg.norm(query_img_vec, axis=1, keepdims=True)
    query_combined = np.concatenate([query_vec, query_img_vec], axis=1)

    # Search
    top_k = 1
    nn_indices, distances = index.get_nns_by_vector(query_combined[0].tolist(), top_k, include_distances=True)
    pids = [str(id_map[idx]) for idx in nn_indices]
    

    # Fetch metadata for all at once
    docs = list(mongo.products.find(
        {"id": {"$in": pids}},   # << filter by Annoy result IDs
        {"_id": 0, "id": 1, "name": 1, "category": 1, "image_url": 1}
    ))

    print("🔍 Query results:")
    # Map by ID for quick lookup
    doc_map = {doc["id"]: doc for doc in docs}

    # Print results in order of Annoy distances
    for idx, dist in zip(nn_indices, distances):
        pid = int(id_map[idx])
        doc = doc_map.get(pid)
        if doc:
            print(f" - ID: {doc['id']}, "
                  f"Name: {doc.get('name')}, "
                  f"Category: {doc.get('category')}, "
                  f"Image: {doc.get('image_url')}, "
                  f"Distance={dist:.4f}")
    
if __name__ == "__main__":
    main()