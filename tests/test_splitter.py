"""Unit tests for rag/splitter.py's header-prefixing fix.

Regression test for a real retrieval bug: MarkdownHeaderTextSplitter
strips a section's header line out of the chunk text into metadata by
default, so a terse section like a one-line bullet list loses the only
words that actually anchor it to a query about that section (see
README's "Retrieval quality" section for the full story: this made
reimbursement_policy.md unretrievable for "What kinds of expenses qualify
for reimbursement?" even with cross-encoder reranking, because the chunk
text itself never contained "qualify" or "reimbursement")."""
from rag.splitter import split_markdown

MD = """# Reimbursement Policy

## What qualifies
Business travel, lodging, meals, and essential supplies.

## Process
1. Keep itemized receipts.
2. Submit within 30 days.
"""


def test_chunk_text_includes_header_path_not_just_metadata():
    docs = split_markdown(MD, source="reimbursement_policy.md")
    qualifies_chunk = next(d for d in docs if "Business travel" in d.page_content)
    # The words that matter for retrieval must be in the TEXT that gets
    # embedded/indexed, not just in metadata — metadata alone doesn't help
    # a bi-encoder or cross-encoder score this chunk against the query.
    assert "What qualifies" in qualifies_chunk.page_content
    assert "Reimbursement Policy" in qualifies_chunk.page_content


def test_metadata_still_has_header_fields_for_citations():
    # The fix adds header text to page_content; it shouldn't remove it
    # from metadata, which tools/doc_search.py and tools/qa_chain.py use
    # to build citations like "(reimbursement_policy.md — What qualifies)".
    docs = split_markdown(MD, source="reimbursement_policy.md")
    qualifies_chunk = next(d for d in docs if "Business travel" in d.page_content)
    assert qualifies_chunk.metadata.get("h1") == "Reimbursement Policy"
    assert qualifies_chunk.metadata.get("h2") == "What qualifies"


def test_source_is_set_on_every_chunk():
    docs = split_markdown(MD, source="reimbursement_policy.md")
    assert docs
    assert all(d.metadata.get("source") == "reimbursement_policy.md" for d in docs)


def test_no_headers_falls_back_to_plain_chunk_text():
    docs = split_markdown("Just a plain paragraph with no headers at all.", source="x.md")
    assert len(docs) == 1
    assert docs[0].page_content == "Just a plain paragraph with no headers at all."
