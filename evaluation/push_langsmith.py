"""Step 3 (optional): publish scored results to LangSmith as an experiment.

    uv run python eval/push_langsmith.py --name v1.2-hybrid-rerank
Requires LANGSMITH_API_KEY.
"""
import argparse, math
import pandas as pd
from langsmith import Client, evaluate
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file if present

DATASET = "agentic-rag-ragas-eval"



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="evaluation/results/per_sample.csv")
    ap.add_argument("--name", required=True, help="experiment prefix, e.g. v1.2-hybrid-rerank")
    args = ap.parse_args()

    import pathlib
    if not pathlib.Path(args.csv).exists():
        print(f"Results CSV not found: {args.csv}. Run score.py first.")
        return

    df = pd.read_csv(args.csv)
    metric_cols = [c for c in df.columns if c not in
                   {"question", "reference", "answer", "contexts", "abstained", "retry_count",
                    "latency_s", "error"}]
    by_q = df.set_index("question").to_dict("index")

    client = Client()
    if not client.has_dataset(dataset_name=DATASET):
        ds = client.create_dataset(DATASET, description="RAGAS eval on rag-mini-wikipedia")
        client.create_examples(
            inputs=[{"question": q} for q in df["question"]],
            outputs=[{"reference": r} for r in df["reference"]],
            dataset_id=ds.id,
        )

    def target(inputs: dict) -> dict:  
        row = by_q.get(inputs["question"], {})
        return {
            "answer": row.get("answer", ""), 
            "abstained": bool(row.get("abstained", False)),
            "retry_count": int(row.get("retry_count", 0)), 
            "latency_s": float(row.get("latency_s", 0.0))
        }

    def make_eval(col):
        def fn(inputs: dict, outputs: dict):
            row = by_q.get(inputs["question"], {})
            v = row.get(col)
            return {"key": col, "score": None if (isinstance(v, float) and math.isnan(v)) else float(v) if v is not None else None}
        fn.__name__ = col
        return fn

    evaluate(target, data=DATASET, evaluators=[make_eval(c) for c in metric_cols],
             experiment_prefix=args.name, max_concurrency=4)


if __name__ == "__main__":
    main()