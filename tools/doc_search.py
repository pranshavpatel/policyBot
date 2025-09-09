from typing import Dict, Any, List, Tuple
from rag.vectorstore import get_retriever

def doc_search(query: str, k: int = 5) -> Dict[str, Any]:
    retriever = get_retriever(k=k)
    docs = retriever.get_relevant_documents(query)

    def sect(md):
        return md.get("h2") or md.get("h1") or md.get("h3") or ""

    return {
        "snippets": [d.page_content for d in docs],
        "citations": [
            {"source": d.metadata.get("source"), "section": sect(d.metadata)} for d in docs
        ],
    }
