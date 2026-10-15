from pathlib import Path
import json
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Sources.rag_core.retriever import TrafficHybridRetriever, RetrievedChunk
from tqdm import tqdm

from test.evaluation.metrics import calculate_recall_at_k

QDRANT_URL = os.getenv("QDRANT_URL") or None
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
DEFAULT_JSONL = (
    PROJECT_ROOT / "Data" / "test" / "evaluation.jsonl"
)

def load_test_dataset(path: str | Path = DEFAULT_JSONL) -> list[dict]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {path}"
        )

    dataset = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {e}"
                ) from e
            dataset.append(sample)

    if not dataset:
        raise ValueError(
            f"Test dataset is empty: {path}"
        )

    return dataset

def evaluate_retriever(
    retriever: TrafficHybridRetriever,
    test_dataset,
    k=10
):

    recalls = []

    for sample in tqdm(
        test_dataset,
        desc="Evaluating Retriever",
        unit="query"
    ):

        query = sample["query"]

        relevant_chunk_ids = sample["relevant_chunks"]

        retrieved_chunks = retriever.get_relevant_chunks(query)

        retrieved_chunk_ids = [
            chunk.metadata["chunk_id"]
            for chunk in retrieved_chunks
        ]

        recall = calculate_recall_at_k(
            relevant_chunk_ids,
            retrieved_chunk_ids,
            k
        )

        recalls.append(recall)

    return sum(recalls) / len(recalls)

def main():

    test_dataset = load_test_dataset()

    retriever = TrafficHybridRetriever(
        qdrant_url=QDRANT_URL,
        qdrant_api_key=QDRANT_API_KEY,
        enable_reranker=True
    )

    recall_at_10 = evaluate_retriever(
        retriever=retriever,
        test_dataset=test_dataset,
        k=10
    )

    print(f"Recall@10: {recall_at_10:.4f}")

if __name__ == "__main__":
    main()
