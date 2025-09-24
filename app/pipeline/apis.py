import sys
import json
import config

from annoy import AnnoyIndex
from db.mongo_client import MongoDBHandler
from models.local_embedder import LocalEmbedder
from models.triton_embedder import TritonEmbedder

def run_search(query_text: str, query_image_url: str, embedder_type: str):
    # Load Annoy index metadata
    with open(config.ID_MAP_PATH, "r", encoding="utf-8") as f:
      data = json.load(f)
      id_map = {int(k): v for k, v in data["id_map"].items()}
      dim = data["dim"]

    index = AnnoyIndex(dim, "angular")
    index.load(config.INDEX_PATH)

    # ---- Init MongoDB ----
    mongo = MongoDBHandler()

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
        {"_id": 0, "id": 1, "name": 1, "category": 1, "price": 1, "image_url": 1}
    ))
    doc_map = {doc["id"]: doc for doc in docs}

    print("🔍 Query results:")
    results = []
    for idx, dist in zip(nn_indices, distances):
        pid = str(id_map[idx])
        doc = doc_map.get(pid)
        if doc:
          result = {
              "id": doc["id"],
              "name": doc.get("name"),
              "category": doc.get("category"),
              "price": doc.get("price"),
              "image_url": doc.get("image_url"),
              "distance": round(dist, 4)
          }
          print(
              f" - ID: {result['id']}, "
              f"Name: {result['name']}, "
              f"Category: {result['category']}, "
              f"Price: {result['price']}, "
              f"Image: {result['image_url']}, "
              f"Distance={result['distance']}"
          )
          results.append(result)

    return {"results": results}
