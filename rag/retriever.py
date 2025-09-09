from typing import List, Dict, Any
import numpy as np
from .embedder import Embedder
from .index import VectorIndex

class Retriever:
    def __init__(self, index: VectorIndex, embedder: Embedder):
        self.index = index
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q_vec = self.embedder.encode([query])[0]  # (D,)
        return self.index.search(q_vec, top_k=top_k)
