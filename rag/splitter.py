from typing import List
from langchain_core.documents import Document
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
        meta = {"source": source}
        meta.update(s.metadata or {})
        # MarkdownHeaderTextSplitter strips header lines out of page_content
        # into metadata (h1/h2/h3) by default — good for citations, bad for
        # retrieval: a short, terse section (e.g. a 1-line bullet list under
        # "## What qualifies") loses the only words that actually anchor it
        # semantically or lexically to a question like "what qualifies for
        # reimbursement?". Found live: this exact case made
        # reimbursement_policy.md unretrievable for that question even with
        # cross-encoder reranking (see README's "Retrieval quality"
        # section) — a bigger/better model didn't help, because the chunk
        # text itself never contained "qualify" or "reimbursement". Fix:
        # re-prepend the header path to the text that actually gets
        # embedded/indexed, not just stored as metadata.
        header_path = " > ".join(meta[h] for h in ("h1", "h2", "h3") if meta.get(h))
        for chunk in body_splitter.split_text(s.page_content):
            text = f"{header_path}\n\n{chunk}" if header_path else chunk
            docs.append(Document(page_content=text, metadata=dict(meta)))
    return docs
