import sys
from pathlib import Path
import pandas as pd
# Adds project root (one level up from 'evaluation/') to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


import asyncio
from dotenv import load_dotenv
from datasets import load_dataset
from ragas import evaluate
from ragas.metrics import (  
    AnswerCorrectness, ContextEntityRecall, Faithfulness, FactualCorrectness,
    LLMContextPrecisionWithReference, LLMContextRecall, NoiseSensitivity,
    ResponseRelevancy, SemanticSimilarity,
)

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from models.required_models import get_chat_model
from langchain_openai import OpenAIEmbeddings

load_dotenv()  # Load environment variables from .env file if present


async def run_huggingface_ragas_evaluation():
    print("Loading evaluation dataset from Hugging Face...")

    hf_dataset = load_dataset("explodinggradients/amnesty_qa", "english_v2", split="eval")

    eval_llm = LangchainLLMWrapper(get_chat_model())
    eval_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    print("Running RAGAS evaluation suite on Hugging Face data...")

    results = evaluate(
        dataset=hf_dataset,
        llm=eval_llm,
        embeddings=eval_embeddings,
        metrics = [
            Faithfulness(llm=eval_llm),
            ResponseRelevancy(llm=eval_llm, embeddings=eval_embeddings),
            LLMContextPrecisionWithReference(llm=eval_llm),
            LLMContextRecall(llm=eval_llm),
            AnswerCorrectness(llm=eval_llm, embeddings=eval_embeddings),
            ]
    )

    # 4. Print results scorecard
    print("\n--- RAGAS Evaluation Scorecard ---")
    df = results.to_pandas()

    columns = [
        "user_input",
        "faithfulness",
        "answer_relevancy",
        "llm_context_precision_with_reference",
        "context_recall",
        "answer_correctness"
    ]

    print(df[columns].to_string(index=False))

    df.to_csv("evaluation/ragas_report.csv", index=False)
    print("\nDetailed report saved to evaluation/ragas_report.csv")


if __name__ == "__main__":
    asyncio.run(run_huggingface_ragas_evaluation())

