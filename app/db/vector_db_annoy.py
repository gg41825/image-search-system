from typing import List, Dict, Any, Optional
from annoy import AnnoyIndex
import numpy as np
from models.bert_embedder import BERTEmbedder

class AnnoyVectorDB:
    """
    Wrapper around Annoy for building and querying a vector index.
    Works on Apple Silicon (M1/M2).
    """

    def __init__(self, num_trees: int = 10, metric: str = "angular", verbose: bool = True):
        """
        Initialize an Annoy index.

        Args:
            num_trees (int): Number of trees to build (trade-off speed vs accuracy).
            metric (str): Distance metric ("angular", "euclidean", "manhattan", etc.).
            verbose (bool): Print debug info.
        """
        self.index: Optional[AnnoyIndex] = None
        self.id_map: Dict[int, str] = {}
        self.num_trees = num_trees
        self.metric = metric
        self.verbose = verbose
        self.dim: Optional[int] = None

    def build_index(self, docs: List[Dict[str, Any]], embedder: BERTEmbedder) -> int:
        """
        Build Annoy index from a list of documents.
        Each document must have: _id, name, category.

        Args:
            docs (List[Dict[str, Any]]): MongoDB-like documents.
            embedder (BERTEmbedder): Text embedder instance.

        Returns:
            int: Embedding dimension.
        """
        if not docs:
            raise ValueError("No documents provided to build_index")

        if self.verbose:
            print(f"[AnnoyVectorDB] Building index with {len(docs)} documents...")

        # Reset state
        self.index = None
        self.id_map.clear()

        # Prepare texts
        texts = [f"{d.get('name')} ({d.get('category')})" for d in docs]
        ids = [str(d.get("id")) for d in docs]

        # Generate embeddings
        txt_emb = embedder.embed_texts(texts)
        vecs = np.copy(txt_emb.detach().cpu().numpy().astype("float32"))

        self.dim = vecs.shape[1]
        self.index = AnnoyIndex(self.dim, self.metric)

        # Add vectors
        for i, v in enumerate(vecs):
            self.index.add_item(i, v.tolist())
            self.id_map[i] = ids[i]

        # Build trees
        self.index.build(self.num_trees)

        if self.verbose:
            print(f"[AnnoyVectorDB] Built index (dim={self.dim}, trees={self.num_trees})")

        return self.dim

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the Annoy index for nearest neighbors.

        Args:
            query_vec (np.ndarray): Query vector, shape [dim] or [1, dim].
            top_k (int): Number of nearest neighbors to return.

        Returns:
            List[Dict[str, Any]]: Search results with mapped IDs and distances.
        """
        if self.index is None or self.dim is None:
            raise RuntimeError("Annoy index has not been built yet")

        if query_vec.ndim == 2:  # [1, dim]
            query_vec = query_vec[0]

        indices, distances = self.index.get_nns_by_vector(
            query_vec.tolist(), top_k, include_distances=True
        )

        results = []
        for rank, (idx, dist) in enumerate(zip(indices, distances)):
            results.append({
                "rank": rank,
                "id": self.id_map.get(idx),
                "distance": float(dist)
            })

        return results
