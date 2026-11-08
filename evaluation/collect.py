"""Step 1: Ingest a public corpus into the running workbench, then collect agent outputs.

    uv run python eval/collect.py ingest --per-file 100
    uv run python eval/collect.py run --n 100 --workers 1
"""
import argparse, json, os, random, re, time, uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from datasets import load_dataset

BASE_URL = os.getenv("RAG_BASE_URL", "http://localhost:8000")
OWNER_HEADER = {os.getenv("RAG_OWNER_HEADER", "X-Owner-Id"): os.getenv("RAG_OWNER_ID", "ragas-eval")}
OUT_DIR = Path("evaluation/results")
THINK = re.compile(r"<think>.*?</think>", re.S)


def call_agent(client: httpx.Client, question: str) -> dict:
    # Matches your FastAPI router structure for chat
    payload = {
        "message": question,
        "thread_id": f"thread_{uuid.uuid4().hex[:8]}"
    }
    r = client.post("/api/chat", json=payload, timeout=300)
    r.raise_for_status()
    d = r.json()
    
    # Extracts retrieved chunks / contexts from your graph state response
    ctx = d.get("citations") or d.get("retrieved_contexts") or d.get("contexts") or []
    ctx_texts = []
    for c in ctx:
        if isinstance(c, str):
            ctx_texts.append(c)
        elif isinstance(c, dict):
            ctx_texts.append(c.get("text") or c.get("page_content") or c.get("content") or "")
            
    answer_text = d.get("response") or d.get("answer") or d.get("generation") or ""
    
    return {
        "answer": THINK.sub("", answer_text).strip(),
        "contexts": [c for c in ctx_texts if c],
        "abstained": bool(d.get("abstained", False)),
        "retry_count": int(d.get("retry_count", 0)),
    }
# -----------------------------------------------------------------------------------------


def ingest(per_file: int):
    passages = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus", split="passages")["passage"]
    with httpx.Client(base_url=BASE_URL, headers=OWNER_HEADER, timeout=600) as c:
        for i in range(0, len(passages), per_file):
            body = "\n\n".join(passages[i : i + per_file]).encode()
            name = f"wiki_{i // per_file:03d}.txt"
            # Matches your file source upload endpoint
            c.post("/api/sources/file", files={"file": (name, body, "text/plain")}).raise_for_status()
            print("ingested", name)


def run(n: int, seed: int, workers: int, out: Path):
    qa = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer", split="test")
    idx = random.Random(seed).sample(range(len(qa)), min(n, len(qa)))
    client = httpx.Client(base_url=BASE_URL, headers=OWNER_HEADER)

    def one(i):
        q, ref = qa[i]["question"], qa[i]["answer"]
        t0 = time.perf_counter()
        try:
            rec = call_agent(client, q)
            rec["error"] = None
        except Exception as e:
            rec = {"answer": "", "contexts": [], "abstained": False, "retry_count": 0, "error": str(e)}
        rec.update(question=q, reference=ref, latency_s=round(time.perf_counter() - t0, 2))
        return rec

    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(one, idx))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in rows))
    print(f"wrote {len(rows)} rows -> {out}  (errors: {sum(bool(r['error']) for r in rows)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ingest"); a.add_argument("--per-file", type=int, default=100)
    b = sub.add_parser("run")
    b.add_argument("--n", type=int, default=50); b.add_argument("--seed", type=int, default=42)
    b.add_argument("--workers", type=int, default=1); b.add_argument("--out", type=Path, default=OUT_DIR / "responses.jsonl")
    args = ap.parse_args()
    ingest(args.per_file) if args.cmd == "ingest" else run(args.n, args.seed, args.workers, args.out)