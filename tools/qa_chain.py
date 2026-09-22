from typing import Tuple, List
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from config import GROQ_API_KEY, GROQ_MODEL
from rag.vectorstore import get_retriever

SYSTEM_PROMPT = """
You are an HR policy assistant. Answer ONLY using the provided context.
- If you state a number or date, quote it verbatim.
- Keep answers concise (1-3 sentences).
- Add a final line: "Source: <source> — <section>" if available.
- If the answer is not in context, say: "I don’t have that in the policy."
Context:
{context}
Question: {question}
Answer:
"""


class QAChain:
    """Minimal 'stuff' RAG chain: retrieve -> stuff context into prompt -> LLM.

    Replaces the legacy langchain.chains.RetrievalQA, which is incompatible
    with the langchain-core version this project resolves to (its Chain
    base class imports langchain_core.memory.BaseMemory, removed upstream).
    A hand-rolled LCEL-style chain has no such dependency and is the
    currently-recommended pattern anyway.
    """

    def __init__(self, llm: ChatGroq, retriever, prompt: PromptTemplate):
        self.llm = llm
        self.retriever = retriever
        self.prompt = prompt

    def invoke(self, inputs: dict) -> dict:
        query = inputs["query"]
        docs: List[Document] = self.retriever.invoke(query)
        context = "\n\n".join(d.page_content for d in docs)
        message = self.llm.invoke(self.prompt.format(context=context, question=query))
        return {"result": message.content, "source_documents": docs}


def build_qa_chain(k: int = 5):
    llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0)
    retriever = get_retriever(k=k)
    prompt = PromptTemplate(
        template=SYSTEM_PROMPT.strip(),
        input_variables=["context", "question"],
    )
    return QAChain(llm, retriever, prompt)


def ask(qa: QAChain, query: str) -> Tuple[str, List[Document]]:
    out = qa.invoke({"query": query})
    return out["result"].strip(), out["source_documents"]
