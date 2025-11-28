import sys
import json
import config

import asyncio
from annoy import AnnoyIndex
from db.mongo_client import MongoDBHandler
from models.local_embedder import LocalEmbedder
from models.triton_embedder import TritonEmbedder

async def run_search(query_text: str, query_image_url: str, embedder_type: str):
    """
    Performs vector similarity search on the Annoy index using a combined 
    text/image query.
    
    Args:
        query_text (str): Text input for the search.
        query_image_url (str): URL or local path of the query image.
        embedder_type (str): Specifies the embedding source ('local' or 'triton').
        
    Returns:
        Dict: A dictionary containing the search results.
    """
    async def load_index_data():
        # 1. Load Annoy index metadata (Blocking File I/O)
        with open(config.ID_MAP_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            id_map = {int(k): v for k, v in data["id_map"].items()}
            dim = data["dim"]

        # 2. Load the Annoy index from disk (Blocking File I/O + Compute)
        index = AnnoyIndex(dim, "angular")
        index.load(config.INDEX_PATH)
        return index, dim, id_map
    # Run the file loading synchronously in a thread and await the result
    index, dim, id_map = await asyncio.to_thread(load_index_data)

    # Init MongoDB handler
    mongo = MongoDBHandler()

    # Choose embedder
    if not query_text and not query_image_url:
        print("You must provide at least --query_text or --image_url")
        sys.exit(1)

    # Function to run the embedding logic synchronously in a thread
    async def get_embeddings():
        if embedder_type == "local":
            print("Using Local Embedder")
            embedder = LocalEmbedder()
            return embedder.embed(
                texts=[query_text] if query_text else None,
                images=[query_image_url] if query_image_url else None
            )

        elif embedder_type == "triton":
            print("Using Triton Embedder (remote inference)")
            # Triton API calls are usually network I/O, but we treat it as blocking here 
            # unless the Triton client itself is async.
            triton = TritonEmbedder(url=config.TRITON_URL, model_name="aligned")
            return triton.embed(query_text, query_image_url)

        else:
            raise ValueError("embedder_type must be 'local' or 'triton'")
    query_combined = await asyncio.to_thread(get_embeddings)

    # Search for the nearest neighbors
    top_k = 1 # Only find the best match
    nn_indices, distances = index.get_nns_by_vector(
        query_combined[0].tolist(), top_k, include_distances=True
    )
    pids = [str(id_map[idx]) for idx in nn_indices]

    # Fetch product metadata from MongoDB
    async def fetch_mongo_products(pids_list):
        # Fetch product metadata from MongoDB (Blocking Network I/O)
        return list(mongo.products.find(
            {"id": {"$in": pids_list}},
            {"_id": 0, "id": 1, "name": 1, "category": 1, "image_url": 1}
        ))
        
    docs = await asyncio.to_thread(fetch_mongo_products, pids)

    doc_map = {doc["id"]: doc for doc in docs}
    print("Query results:")
    results = []
    for idx, dist in zip(nn_indices, distances):
        pid = str(id_map[idx])
        doc = doc_map.get(pid)
        if doc:
          result = {
              "id": doc["id"],
              "name": doc.get("name"),
              "category": doc.get("category"),
              "image_url": doc.get("image_url"),
              "distance": round(dist, 4)
          }
          print(
              f" - ID: {result['id']}, "
              f"Name: {result['name']}, "
              f"Category: {result['category']}, "
              f"Image: {result['image_url']}, "
              f"Distance={result['distance']}"
          )
          results.append(result)

    return {"results": results}