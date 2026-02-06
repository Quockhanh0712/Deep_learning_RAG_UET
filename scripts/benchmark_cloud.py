"""
Qdrant Cloud Benchmark Script.
Measures: Latency, Recall@5, Throughput for Legal RAG queries.
Generates comparison report against expected keywords.
"""
import sys
import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import settings
from backend.core.qdrant_store import get_qdrant_connector
from backend.core.embeddings import get_embedding_model

# Paths
QUERIES_FILE = os.path.join(os.path.dirname(__file__), "..", "tests", "queries", "legal_test_queries.json")
REPORT_FILE = os.path.join(os.path.dirname(__file__), "..", "QDRANT_CLOUD_BENCHMARK_RESULTS.md")


def load_queries() -> List[Dict[str, Any]]:
    """Load test queries from JSON file."""
    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_recall(results: List[Dict], expected_keywords: List[str]) -> float:
    """
    Calculate Recall@5: How many expected keywords appear in top 5 results?
    Returns percentage (0-100).
    """
    if not results or not expected_keywords:
        return 0.0
    
    # Combine text from top 5 results
    combined_text = " ".join([
        r.get("text", "") + " " + str(r.get("metadata", {}))
        for r in results[:5]
    ]).lower()
    
    # Check how many keywords are found
    found = sum(1 for kw in expected_keywords if kw.lower() in combined_text)
    return (found / len(expected_keywords)) * 100


def run_benchmark():
    """Run full benchmark suite."""
    print("\n" + "=" * 70)
    print("🚀 QDRANT CLOUD PERFORMANCE BENCHMARK".center(70))
    print("=" * 70)
    print(f"Host: {settings.QDRANT_HOST}")
    print(f"Collection: {settings.QDRANT_LEGAL_COLLECTION}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")
    
    # Load components
    try:
        queries = load_queries()
        print(f"📋 Loaded {len(queries)} test queries")
    except Exception as e:
        print(f"❌ Failed to load queries: {e}")
        return
    
    try:
        connector = get_qdrant_connector()
        embedder = get_embedding_model()
        print("✅ Embedder and Qdrant connector initialized")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    # Benchmark results
    results_data = []
    total_embed_time = 0
    total_search_time = 0
    
    print("\n📊 Running benchmarks...\n")
    
    for q in queries:
        query_id = q["id"]
        query_text = q["query"]
        expected_keywords = q["expected_keywords"]
        
        print(f"[{query_id:02d}] {query_text[:40]}...", end=" ", flush=True)
        
        try:
            # Embedding
            embed_start = time.time()
            query_vector = embedder.embed([query_text])[0].tolist()
            embed_time = (time.time() - embed_start) * 1000
            total_embed_time += embed_time
            
            # Search
            search_start = time.time()
            search_results = connector.hybrid_search(
                query_vector=query_vector,
                top_k=5,
                collection="legal"
            )
            search_time = (time.time() - search_start) * 1000
            total_search_time += search_time
            
            # Calculate Recall
            recall = calculate_recall(search_results, expected_keywords)
            
            # Total latency
            total_time = embed_time + search_time
            
            # Status
            status = "✅ PASS" if recall >= 50 else "⚠️ LOW RECALL" if recall > 0 else "❌ FAIL"
            
            results_data.append({
                "id": query_id,
                "query": query_text,
                "embed_ms": round(embed_time),
                "search_ms": round(search_time),
                "total_ms": round(total_time),
                "recall": round(recall),
                "results_count": len(search_results),
                "status": status
            })
            
            print(f"⏱️ {total_time:.0f}ms | Recall: {recall:.0f}% | {status}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results_data.append({
                "id": query_id,
                "query": query_text,
                "embed_ms": -1,
                "search_ms": -1,
                "total_ms": -1,
                "recall": 0,
                "results_count": 0,
                "status": "❌ ERROR"
            })
    
    # Summary Statistics
    print("\n" + "=" * 70)
    print("📈 BENCHMARK SUMMARY".center(70))
    print("=" * 70)
    
    successful = [r for r in results_data if r["total_ms"] > 0]
    
    if successful:
        avg_embed = sum(r["embed_ms"] for r in successful) / len(successful)
        avg_search = sum(r["search_ms"] for r in successful) / len(successful)
        avg_total = sum(r["total_ms"] for r in successful) / len(successful)
        avg_recall = sum(r["recall"] for r in successful) / len(successful)
        
        print(f"📊 Queries Tested: {len(successful)}/{len(queries)}")
        print(f"⏱️ Avg Embed Time: {avg_embed:.0f}ms")
        print(f"⏱️ Avg Search Time: {avg_search:.0f}ms")
        print(f"⏱️ Avg Total Latency: {avg_total:.0f}ms")
        print(f"🎯 Avg Recall@5: {avg_recall:.1f}%")
        
        # Throughput (queries per second)
        total_time_sec = (total_embed_time + total_search_time) / 1000
        throughput = len(successful) / total_time_sec if total_time_sec > 0 else 0
        print(f"🚀 Throughput: {throughput:.2f} queries/sec")
    else:
        print("❌ No successful queries to analyze")
        avg_embed = avg_search = avg_total = avg_recall = throughput = 0
    
    # Generate Markdown Report
    generate_report(results_data, {
        "avg_embed_ms": round(avg_embed) if successful else 0,
        "avg_search_ms": round(avg_search) if successful else 0,
        "avg_total_ms": round(avg_total) if successful else 0,
        "avg_recall": round(avg_recall, 1) if successful else 0,
        "throughput": round(throughput, 2) if successful else 0,
        "total_queries": len(queries),
        "successful_queries": len(successful)
    })
    
    print(f"\n📄 Report saved to: {REPORT_FILE}")
    print("=" * 70)


def generate_report(results: List[Dict], stats: Dict):
    """Generate Markdown benchmark report."""
    report = f"""# Qdrant Cloud Benchmark Results

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Host**: `{settings.QDRANT_HOST}`  
**Collection**: `{settings.QDRANT_LEGAL_COLLECTION}`

---

## 📊 Summary Statistics

| Metric | Value |
|--------|-------|
| **Total Queries** | {stats['total_queries']} |
| **Successful** | {stats['successful_queries']} |
| **Avg Embed Time** | {stats['avg_embed_ms']}ms |
| **Avg Search Time** | {stats['avg_search_ms']}ms |
| **Avg Total Latency** | {stats['avg_total_ms']}ms |
| **Avg Recall@5** | {stats['avg_recall']}% |
| **Throughput** | {stats['throughput']} q/s |

---

## 📋 Detailed Results

| ID | Query | Embed (ms) | Search (ms) | Total (ms) | Recall@5 | Status |
|----|-------|------------|-------------|------------|----------|--------|
"""
    
    for r in results:
        query_short = r["query"][:30] + "..." if len(r["query"]) > 30 else r["query"]
        report += f"| {r['id']} | {query_short} | {r['embed_ms']} | {r['search_ms']} | {r['total_ms']} | {r['recall']}% | {r['status']} |\n"
    
    report += f"""
---

## 🎯 Verdict

"""
    
    if stats['avg_recall'] >= 70 and stats['avg_total_ms'] < 2000:
        report += "✅ **PASS**: Qdrant Cloud performance is excellent. Recall is high and latency is acceptable.\n"
    elif stats['avg_recall'] >= 50:
        report += "⚠️ **ACCEPTABLE**: Recall is moderate. Consider tuning search parameters or reranking.\n"
    else:
        report += "❌ **NEEDS IMPROVEMENT**: Low recall detected. Review embedding model or collection data.\n"
    
    report += f"""
---

## 🔧 Recommendations

1. **If latency is high (>2s)**: Consider upgrading Qdrant Cloud tier or using async embedding.
2. **If recall is low (<70%)**: Check if expected keywords are stored in payload metadata.
3. **For production**: Enable reranker for improved precision.

---

*Report generated by `scripts/benchmark_cloud.py`*
"""
    
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    run_benchmark()
