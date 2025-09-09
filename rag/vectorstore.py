from typing import List
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from .embeddings import get_embeddings
from config import CHROMA_DIR

def upsert_documents(docs: List[Document]):
    """Create or update a persistent Chroma DB from documents."""
    emb = get_embeddings()
    # If the directory exists, this will append; to rebuild, clear the folder first.
    vs = Chroma.from_documents(
        documents=docs,
        embedding=emb,
        persist_directory=CHROMA_DIR,
    )
    vs.persist()

def get_retriever(k: int = 5):
    """Open the persistent Chroma DB and return a retriever."""
    emb = get_embeddings()
    vs = Chroma(
        embedding_function=emb,
        persist_directory=CHROMA_DIR,
    )
    return vs.as_retriever(search_kwargs={"k": k})
