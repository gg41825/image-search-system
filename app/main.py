from db.mongo_client import MongoDBHandler
from db.init_data import seed_products, seed_product_vectors_bert
from models.bert_embedder import BERTEmbedder
import sys

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

    # Fetch product docs
    docs = list(mongo.products.find({}, {"_id": 1, "name": 1, "category": 1}))

    if not docs:
        # Try to seed products if empty
        inserted_ids = seed_products(mongo, force_drop=False)
        if not inserted_ids:
            mongo.log_error(
                error_type="data",
                message="No products inserted during seed_products",
                stacktrace="seed_products returned 0 inserted_ids"
            )
            print("❌ No products available after seeding. Exiting.")
            sys.exit(1)  # stop program safely

        # Refresh docs after seeding
        docs = list(mongo.products.find({}, {"_id": 1, "name": 1, "category": 1}))
        if not docs:
            mongo.log_error(
                error_type="data",
                message="Products still missing after seed_products",
                stacktrace="MongoDB products collection is empty"
            )
            print("❌ Products collection is empty even after seeding. Exiting.")
            sys.exit(1)

    # Seed vector DB safely
    try:
        index, dim, id_map = seed_product_vectors_bert(mongo)
    except Exception as e:
        mongo.log_error(
            error_type="vector_db",
            message="Failed to seed product vectors with bert",
            stacktrace=str(e)
        )
        print(f"❌ Failed to build vector DB: {e}")
        sys.exit(1)

    # Query example
    bert = BERTEmbedder()
    query = bert.embed_texts(["Cocoon - Silk Wash Textilpflege"])[0]
    indices, distances = index.get_nns_by_vector(query, 3, include_distances=True)

    print("\n🔎 Search results:")
    for idx, dist in zip(indices, distances):
        product_id = id_map[idx]
        product = mongo.get_product(product_id)
        if product:
            print(f"- {product['name']} ({product['category']}) | distance={dist:.4f}")
        else:
            print(f"⚠️ Product with _id={product_id} not found in MongoDB")
    

if __name__ == "__main__":
    main()