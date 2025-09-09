from pathlib import Path
from rag.splitter import split_markdown
from rag.vectorstore import upsert_documents
from config_paths import VECTORSTORE_DIR, CHROMA_COLLECTION, EMBED_MODEL

def main():
    data_dir = Path("data/policies")
    paths = list(data_dir.glob("*.md"))
    all_docs = []
    for p in paths:
        text = p.read_text(encoding="utf-8")
        docs = split_markdown(text, source=p.name)
        all_docs.extend(docs)
        print(f"{p.name}: {len(docs)} chunks")
    print(f"Total chunks: {len(all_docs)}. Upserting to Chroma…")
    upsert_documents(all_docs)
    print("Done. Chroma persisted.")

if __name__ == "__main__":
    main()
