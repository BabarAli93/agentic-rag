"""Step 2: score collected responses with RAGAS and write a report.

    uv run python eval/score.py --in eval/results/responses.jsonl --judge gpt-4o-mini [--extended]
"""
import sys
import types

# --- WORKAROUND FOR RAGAS UPSTREAM VERTEX AI IMPORT BUG ---
try:
    import langchain_community.chat_models.vertexai
except (ImportError, ModuleNotFoundError):
    dummy_chat = types.ModuleType("langchain_community.chat_models.vertexai")
    dummy_chat.ChatVertexAI = type("ChatVertexAI", (object,), {})
    sys.modules["langchain_community.chat_models.vertexai"] = dummy_chat
# -----------------------------------------------------------

import argparse, json, warnings
from datetime import date
from pathlib import Path
from dotenv import load_dotenv


import numpy as np
import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (  
    AnswerCorrectness, ContextEntityRecall, Faithfulness, FactualCorrectness,
    LLMContextPrecisionWithReference, LLMContextRecall, NoiseSensitivity,
    ResponseRelevancy, SemanticSimilarity,
)
from ragas.run_config import RunConfig

warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()  # Load environment variables from .env file if present


def boot_ci(s: pd.Series, n=2000, seed=0):
    x = s.dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    means = np.random.default_rng(seed).choice(x, (n, len(x))).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return x.mean(), lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=Path("evaluation/results/responses.jsonl"))
    ap.add_argument("--outdir", type=Path, default=Path("evaluation/results"))
    ap.add_argument("--judge", default="gpt-4o-mini")  
    ap.add_argument("--extended", action="store_true")
    args = ap.parse_args()

    if not args.inp.exists():
        print(f"Input file not found: {args.inp}. Run collect.py first.")
        return

    rows = [json.loads(l) for l in args.inp.read_text().splitlines()]
    n_err = sum(bool(r.get("error")) for r in rows)
    rows = [r for r in rows if not r.get("error")]
    meta = pd.DataFrame(rows)

    ds = EvaluationDataset(samples=[
        SingleTurnSample(
            user_input=r["question"], response=r["answer"] or "(no answer)",
            reference=r["reference"], retrieved_contexts=r["contexts"] or ["(no context retrieved)"],
        ) for r in rows
    ])

    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", temperature=0))
    emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
    metrics = [
        Faithfulness(llm=llm),
        ResponseRelevancy(llm=llm, embeddings=emb),
        LLMContextPrecisionWithReference(llm=llm),
        LLMContextRecall(llm=llm),
        AnswerCorrectness(llm=llm, embeddings=emb),
    ]
    if args.extended:
        metrics += [NoiseSensitivity(llm=llm), ContextEntityRecall(llm=llm),
                    FactualCorrectness(llm=llm), SemanticSimilarity(embeddings=emb)]

    result = evaluate(ds, metrics=metrics, run_config=RunConfig(timeout=180, max_workers=8),
                      raise_exceptions=False, show_progress=True)
    scores = result.to_pandas()
    metric_cols = [c for c in scores.columns if c not in
                   {"user_input", "retrieved_contexts", "response", "reference"}]
    df = pd.concat([meta.reset_index(drop=True), scores[metric_cols].reset_index(drop=True)], axis=1)

    def table(frame):
        out = {}
        for c in metric_cols:
            m, lo, hi = boot_ci(frame[c])
            out[c] = {"mean": m, "ci95_low": lo, "ci95_high": hi, "n_valid": int(frame[c].notna().sum())}
        return out

    answered = df[~df["abstained"]] if "abstained" in df.columns else df
    summary = {
        "date": str(date.today()), "judge": args.judge, "n_scored": len(df), "n_errors": n_err,
        "abstention_rate": float(df["abstained"].mean()) if "abstained" in df.columns else 0.0,
        "avg_retries": float(df["retry_count"].mean()) if "retry_count" in df.columns else 0.0,
        "latency_p50_s": float(df["latency_s"].quantile(0.5)) if "latency_s" in df.columns else 0.0,
        "latency_p95_s": float(df["latency_s"].quantile(0.95)) if "latency_s" in df.columns else 0.0,
        "all": table(df), "answered_only": table(answered),
    }

    args.outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.outdir / "per_sample.csv", index=False)
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    md = [f"# RAGAS report ({summary['date']})",
          f"Judge: `{args.judge}` | scored: {len(df)} | errors: {n_err} | "
          f"abstention: {summary['abstention_rate']:.1%} | latency p50/p95: "
          f"{summary['latency_p50_s']:.1f}s / {summary['latency_p95_s']:.1f}s\n"]
    for title, key in [("All questions", "all"), ("Answered only (excl. abstentions)", "answered_only")]:
        md += [f"## {title}", "| Metric | Mean | 95% CI | n |", "|---|---|---|---|"]
        md += [f"| {c} | {v['mean']:.3f} | [{v['ci95_low']:.3f}, {v['ci95_high']:.3f}] | {v['n_valid']} |"
               for c, v in summary[key].items()]
        md.append("")
    
    if "faithfulness" in df.columns:
        worst = df.sort_values("faithfulness").head(10)[["question", "answer", "faithfulness"]]
        md += ["## 10 least-faithful answers (for error analysis)", worst.to_markdown(index=False)]
        
    (args.outdir / "report.md").write_text("\n".join(md))
    print("\n".join(md[:14]))


if __name__ == "__main__":
    main()