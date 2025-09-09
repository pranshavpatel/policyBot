from typing import Tuple, List
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
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

def build_qa_chain(k: int = 5):
    llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0)
    retriever = get_retriever(k=k)

    prompt = PromptTemplate(
        template=SYSTEM_PROMPT.strip(),
        input_variables=["context", "question"],
    )

    # "stuff" is fine for small contexts; switch to "map_reduce" if docs grow large
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return qa

def ask(qa, query: str) -> Tuple[str, List[Document]]:
    out = qa.invoke({"query": query})
    return out["result"].strip(), out["source_documents"]
