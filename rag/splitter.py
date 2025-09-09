from typing import List
from langchain.schema import Document
# NEW import location:
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

def split_markdown(md_text: str, source: str) -> List[Document]:
    headers = [("#", "h1"), ("##", "h2"), ("###", "h3")]
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)  # <-- changed kwarg
    sections = header_splitter.split_text(md_text)

    body_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=120, separators=["\n\n", "\n", " ", ""]
    )

    docs: List[Document] = []
    for s in sections:
        for chunk in body_splitter.split_text(s.page_content):
            meta = {"source": source}
            meta.update(s.metadata or {})
            docs.append(Document(page_content=chunk, metadata=meta))
    return docs
