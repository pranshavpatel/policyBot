from tools.qa_chain import build_qa_chain, ask

def demo():
    qa = build_qa_chain(k=5)
    qs = [
        "How many PTO days in Year 1?",
        "How to request leave?",
        "Can I carry over PTO to next year?",
    ]
    for q in qs:
        ans, srcs = ask(qa, q)
        print("\nQ:", q)
        print("A:", ans)
        cits = [(d.metadata.get("source"),
                 d.metadata.get("h2") or d.metadata.get("h1") or d.metadata.get("h3", ""))
                for d in srcs]
        print("Citations:", cits[:3])

if __name__ == "__main__":
    demo()
