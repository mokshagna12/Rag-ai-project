from rag import generate_answer
from utils import now_ms, timer_step

def main():
    print("Local RAG Q&A (CLI) — type 'exit' to quit.")
    while True:
        q = input("\n> Question: ").strip()
        if q.lower() in {"exit","quit"}:
            break
        t0 = now_ms()
        out = generate_answer(q)
        t_total = timer_step(t0)
        print("\nAnswer:\n", out["answer"])
        print("\nCitations:", out["citations"])
        print(f"Latency: {t_total} ms")

if __name__ == "__main__":
    main()
