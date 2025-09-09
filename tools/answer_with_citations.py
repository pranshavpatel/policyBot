# tools/answer_with_citations.py
from typing import Dict, Any, List

def synthesize_answer(query: str, hits: List[Dict[str, Any]]) -> str:
    """
    Very simple: extracts the most relevant line(s) and formats a short answer.
    Replace this later with an LLM call that uses the hits as context.
    """
    if not hits:
        return "I couldn’t find anything relevant in the documents."

    top = hits[0]
    text = top["text"]
    meta = top["meta"]
    # For demos, return the exact line that likely answers the question:
    # (In practice, use an LLM with a grounding prompt.)
    answer_line = text.split("\n")[0][:240]  # quick conservative slice

    cite = f" ({meta.get('source')} — {meta.get('section')})" if meta.get("section") else f" ({meta.get('source')})"
    return f"{answer_line}{cite}"
