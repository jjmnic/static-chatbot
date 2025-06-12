# vector_store.py
import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)

    def add_embeddings(self, embeddings: np.ndarray):
        self.index.add(embeddings)

    def search(self, query_embedding: np.ndarray, top_k=5):
        distances, indices = self.index.search(np.array([query_embedding]), top_k)
        return distances, indices