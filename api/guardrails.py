"""
guardrails.py
-------------
Lightweight, dependency-free guardrails for a RAG pipeline.

Two layers, matching the two places things can go wrong:

  InputGuardrail  - runs on the user's query, BEFORE retrieval/LLM call.
                     Blocks prompt-injection attempts, out-of-scope topics,
                     malformed input, and redacts PII before it is logged
                     or sent to the LLM/vector DB.

  OutputGuardrail - runs on the LLM's answer, BEFORE it is returned to the
                     user. Redacts any PII the model may have echoed back,
                     and blocks answers that aren't actually grounded in the
                     retrieved context (a fast heuristic - see note below).

These are intentionally implemented with stdlib `re` rather than an external
service so they run synchronously, in-process, with near-zero latency on
every request. For production you would typically layer these with (or
replace pieces of them with) something like:
  - Microsoft Presidio for proper PII detection/anonymization
  - guardrails-ai / NeMo Guardrails for policy-as-code style rails
  - A moderation endpoint (e.g. Ollama/OpenAI moderation model) for toxicity
This module gives you the same *shape* of protection without adding a new
service dependency, and can be swapped out piece by piece later.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional
import structlog


@dataclass
class GuardrailResult:
    passed: bool
    reason: Optional[str] = None
    # Sanitized/redacted version of the text (query or answer). Callers
    # should use this instead of the original text when passed=True and
    # flags is non-empty (e.g. PII was found and redacted but wasn't
    # severe enough to block the request outright).
    redacted_text: Optional[str] = None
    flags: List[str] = field(default_factory=list)


class InputGuardrail:
    """Validates and sanitizes a user query before it touches retrieval/LLM."""

    PROMPT_INJECTION_PATTERNS = [
        r"ignore (all )?(previous|above|prior) instructions",
        r"disregard (all )?(previous|above|prior) instructions",
        r"you are now (a|an)",
        r"reveal (your|the) (system )?(prompt|instructions)",
        r"print (your|the) (system )?prompt",
        r"jailbreak",
        r"pretend (you are|to be) (?!.*assistant)",
        r"do anything now",
        r"\bDAN\b",
    ]

    PII_PATTERNS = {
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "phone": r"\b(\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b(?:\d[ -]*?){13,16}\b",
    }

    def __init__(
        self,
        max_length: int = 2000,
        min_length: int = 3,
        blocklist_topics: Optional[List[str]] = None,
    ):
        self.logger = structlog.get_logger()
        self.max_length = max_length
        self.min_length = min_length
        self.blocklist_topics = [t.lower() for t in (blocklist_topics or [])]

    def _check_length(self, query: str) -> Optional[str]:
        stripped = query.strip()
        if len(stripped) < self.min_length:
            return "query_too_short"
        if len(query) > self.max_length:
            return "query_too_long"
        return None

    def _check_prompt_injection(self, query: str) -> bool:
        lowered = query.lower()
        return any(re.search(p, lowered) for p in self.PROMPT_INJECTION_PATTERNS)

    def _check_topic_blocklist(self, query: str) -> bool:
        if not self.blocklist_topics:
            return False
        lowered = query.lower()
        return any(topic in lowered for topic in self.blocklist_topics)

    def _detect_pii(self, text: str) -> List[str]:
        found = []
        for label, pattern in self.PII_PATTERNS.items():
            if re.search(pattern, text):
                found.append(label)
        return found

    def _redact_pii(self, text: str) -> str:
        redacted = text
        for label, pattern in self.PII_PATTERNS.items():
            redacted = re.sub(pattern, f"[REDACTED_{label.upper()}]", redacted)
        return redacted

    def validate(self, query: str) -> GuardrailResult:
        length_issue = self._check_length(query)
        if length_issue:
            return GuardrailResult(passed=False, reason=length_issue, flags=[length_issue])

        if self._check_prompt_injection(query):
            self.logger.warning(
                "guardrail_blocked", layer="input", reason="prompt_injection",
                query_preview=query[:80],
            )
            return GuardrailResult(
                passed=False, reason="prompt_injection_detected", flags=["prompt_injection"]
            )

        if self._check_topic_blocklist(query):
            self.logger.warning("guardrail_blocked", layer="input", reason="out_of_scope_topic")
            return GuardrailResult(
                passed=False, reason="out_of_scope_topic", flags=["blocked_topic"]
            )

        pii_found = self._detect_pii(query)
        redacted = query
        flags = []
        if pii_found:
            flags = [f"pii_{p}" for p in pii_found]
            redacted = self._redact_pii(query)
            self.logger.info("pii_redacted", layer="input", types=pii_found)

        return GuardrailResult(passed=True, redacted_text=redacted, flags=flags)


class OutputGuardrail:
    """Validates the LLM's answer before it is returned to the caller."""

    REFUSAL_PHRASES = [
        "i don't have enough information",
        "i do not have enough information",
        "i cannot answer",
        "i don't know",
    ]

    def __init__(self, groundedness_threshold: float = 0.35):
        self.logger = structlog.get_logger()
        self._pii_helper = InputGuardrail()  # reuse the same PII regexes
        self.groundedness_threshold = groundedness_threshold

    def _groundedness_score(self, answer: str, context_chunks: List[str]) -> float:
        """
        Fast, synchronous groundedness heuristic: fraction of the answer's
        vocabulary that also appears in the retrieved context.

        This is deliberately cheap (no extra LLM call, no network round trip)
        so it can run on the hot path for every single response. It is a
        coarse proxy, not a substitute for RAGAS's `faithfulness` metric,
        which uses an LLM judge to check each claim against the context and
        is far more accurate - but also far more expensive/slow. The intended
        split is:
            - this heuristic -> blocks obviously ungrounded answers in real time
            - RAGAS faithfulness -> measures true groundedness offline/in eval
        """
        if not context_chunks:
            return 0.0
        answer_tokens = set(re.findall(r"\w+", answer.lower()))
        if not answer_tokens:
            return 0.0
        context_tokens = set(re.findall(r"\w+", " ".join(context_chunks).lower()))
        overlap = answer_tokens & context_tokens
        return len(overlap) / len(answer_tokens)

    def validate(self, answer: str, context_chunks: List[str]) -> GuardrailResult:
        pii_found = self._pii_helper._detect_pii(answer)
        flags = []
        redacted = answer
        if pii_found:
            flags = [f"pii_{p}" for p in pii_found]
            redacted = self._pii_helper._redact_pii(answer)
            self.logger.warning("pii_redacted", layer="output", types=pii_found)

        is_refusal = any(phrase in answer.lower() for phrase in self.REFUSAL_PHRASES)
        if not is_refusal:
            score = self._groundedness_score(answer, context_chunks)
            if score < self.groundedness_threshold:
                flags.append("low_groundedness")
                self.logger.warning(
                    "guardrail_blocked", layer="output", reason="low_groundedness",
                    score=round(score, 3), threshold=self.groundedness_threshold,
                )
                return GuardrailResult(
                    passed=False,
                    reason="answer_not_grounded_in_context",
                    redacted_text=(
                        "I don't have enough grounded information in the retrieved "
                        "documents to answer that confidently."
                    ),
                    flags=flags,
                )

        return GuardrailResult(passed=True, redacted_text=redacted, flags=flags)