from typing import List, Dict, Any
import numpy as np

class VectorIndex:
    def __init__(self):
        self.ids: List[str] = []
        self.vecs: np.ndarray | None = None
        self.metas: List[Dict[str, Any]] = []
        self.texts: List[str] = []

    def add(self, ids: List[str], vecs: np.ndarray, metas: List[Dict[str, Any]], texts: List[str]):
        assert len(ids) == len(vecs) == len(metas) == len(texts)
        if self.vecs is None:
            self.vecs = vecs
        else:
            self.vecs = np.vstack([self.vecs, vecs])
        self.ids.extend(ids)
        self.metas.extend(metas)
        self.texts.extend(texts)

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        # query_vec shape: (D,) and self.vecs normalized -> cosine = dot
        sims = self.vecs @ (query_vec / (np.linalg.norm(query_vec) + 1e-12))
        idxs = np.argpartition(-sims, top_k)[:top_k]
        idxs = idxs[np.argsort(-sims[idxs])]
        results = []
        for i in idxs:
            results.append({
                "id": self.ids[i],
                "score": float(sims[i]),
                "meta": self.metas[i],
                "text": self.texts[i],
            })
        return results
