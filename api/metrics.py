from prometheus_client import Counter, Histogram

# Counters
RAG_QUERIES_TOTAL = Counter(
    "rag_queries_total",
    "Total number of RAG queries received"
)

# Histograms - with useful labels
TOTAL_QUERY_LATENCY = Histogram(
    "rag_total_query_latency_seconds",
    "End-to-end latency for a /ask request (retrieval + generation)",
    ["backend", "model"],          # Added model label for better insights
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0]  # Good buckets for LLM queries
)