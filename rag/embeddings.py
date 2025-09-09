from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBED_MODEL

def get_embeddings():
    # Local / CPU-friendly; free
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)
