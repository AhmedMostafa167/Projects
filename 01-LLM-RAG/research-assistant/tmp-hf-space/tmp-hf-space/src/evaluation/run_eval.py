"""RAGAS-based evaluation runner.

Computes faithfulness, answer relevancy, context precision, context
recall — the four core RAG quality metrics — against a JSONL dataset of
{question, ground_truth} pairs.

Usage:
    pip install -e ".[eval]"
    python -m src.evaluation.run_eval --dataset data/eval_questions.jsonl

Output: prints metrics and writes eval_results/<timestamp>.json
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.pipeline import ResearchPipeline
from src.utils import get_logger

log = get_logger(__name__)


def _load_dataset(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("eval_results"))
    args = parser.parse_args()

    rows = _load_dataset(args.dataset)
    log.info("loaded_eval_set", n=len(rows))

    pipeline = ResearchPipeline()

    results = []
    for row in rows:
        result = pipeline.ask(row["question"])
        results.append(
            {
                "question": row["question"],
                "ground_truth": row.get("ground_truth", ""),
                "answer": result.answer,
                "contexts": [s["title"] or s["url"] for s in result.sources],
            }
        )

    # Only import ragas when actually evaluating, so the module imports
    # cleanly even without the [eval] extras installed.
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as e:
        raise SystemExit(
            "RAGAS is not installed. Install with: pip install -e \".[eval]\""
        ) from e

    ds = Dataset.from_list(results)
    scores = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out_dir / f"{ts}.json"
    with out_path.open("w") as f:
        json.dump(
            {"scores": dict(scores), "samples": results},
            f,
            indent=2,
        )
    log.info("eval_done", out=str(out_path), scores=dict(scores))
    print(json.dumps(dict(scores), indent=2))


if __name__ == "__main__":
    main()
