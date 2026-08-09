"""
evaluation.py
-------------
Offline / batch quality evaluation of the RAG pipeline using RAGAS.

This is NOT the same job as guardrails.OutputGuardrail:

  - OutputGuardrail runs synchronously, on every single live request, and
    only has time for a cheap heuristic. Its job is to catch obviously bad
    answers before a user ever sees them.

  - RagasEvaluator runs offline/in batch (e.g. nightly, in CI, after a
    prompt or model change) against a curated set of questions - optionally
    with ground-truth answers - and uses an LLM-as-judge to score the
    system on standard RAG metrics. Its job is to tell you, with numbers,
    whether the pipeline is actually getting better or worse over time.

Metrics used:
  - faithfulness:        Does the answer only contain claims supported by
                          the retrieved context? (hallucination check)
  - answer_relevancy:    Does the answer actually address the question?
  - context_precision:   Of the chunks retrieved, how many were relevant?
                          (are you feeding the LLM noise?)
  - context_recall:      Of the chunks that *should* have been retrieved
                          (per a ground-truth answer), how many were?
                          Requires ground_truth, so only computed when
                          ground_truth is supplied.
"""

from typing import Any, Dict, List, Optional
import structlog

try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    RAGAS_AVAILABLE = True
except ImportError:  # pragma: no cover
    RAGAS_AVAILABLE = False


class RagasEvaluator:
    def __init__(self, llm=None, embeddings=None):
        """
        llm / embeddings: optional RAGAS-wrapped LLM/embedding objects
        (e.g. from `ragas.llms.LangchainLLMWrapper`) used as the judge model.
        If omitted, RAGAS falls back to its configured default (typically
        expects an OPENAI_API_KEY). For a fully local stack matching this
        project (Ollama), wrap an Ollama chat model with LangChain's
        `ChatOllama` and pass it in via `LangchainLLMWrapper`.
        """
        self.logger = structlog.get_logger()
        if not RAGAS_AVAILABLE:
            self.logger.warning("ragas_not_installed")
        self.llm = llm
        self.embeddings = embeddings

    def evaluate_batch(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        questions:     list of user questions
        answers:       the RAG pipeline's answer for each question
        contexts:      the retrieved (and, ideally, re-ranked) chunks used
                       for each answer - one List[str] per question
        ground_truths: optional reference answers, enables context_recall

        Returns a dict with an aggregate `summary` and a `per_sample`
        breakdown (one row per question) for drilling into failures.
        """
        if not RAGAS_AVAILABLE:
            raise ImportError("Install with: pip install ragas datasets")

        assert len(questions) == len(answers) == len(contexts), (
            "questions, answers, and contexts must be the same length"
        )

        data = {"question": questions, "answer": answers, "contexts": contexts}
        metrics = [faithfulness, answer_relevancy, context_precision]

        if ground_truths:
            data["ground_truth"] = ground_truths
            metrics.append(context_recall)

        dataset = Dataset.from_dict(data)

        kwargs = {}
        if self.llm:
            kwargs["llm"] = self.llm
        if self.embeddings:
            kwargs["embeddings"] = self.embeddings

        result = evaluate(dataset=dataset, metrics=metrics, **kwargs)
        df = result.to_pandas()

        summary = {
            "faithfulness": float(df["faithfulness"].mean()),
            "answer_relevancy": float(df["answer_relevancy"].mean()),
            "context_precision": float(df["context_precision"].mean()),
        }
        if ground_truths:
            summary["context_recall"] = float(df["context_recall"].mean())

        self.logger.info("ragas_evaluation_completed", **summary)
        return {"summary": summary, "per_sample": df.to_dict(orient="records")}