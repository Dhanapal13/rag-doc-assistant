"""
Prometheus metric definitions for the RAG service.
Defined ONCE here and imported wherever needed — prometheus_client raises
a "Duplicated timeseries" error if you call Counter()/Histogram() twice
for the same metric name (e.g. once in rag_service.py and again in main.py).
"""
from prometheus_client import Counter, Histogram

# --- Counters: count discrete events ---

RAG_QUERIES_TOTAL = Counter(
    "rag_queries_total",
    "Total number of RAG queries received"
)


# --- Histograms: measure how long things take (and let you compute p50/p95/p99) ---

TOTAL_QUERY_LATENCY = Histogram(
    "rag_total_query_latency_seconds",
    "End-to-end latency for a /ask request (retrieval + generation)",
    ["backend"],
)
