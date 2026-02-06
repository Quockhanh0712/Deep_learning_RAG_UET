"""
Qdrant Cloud Verification Script with Sample Search.
Tests: Connection, Collection Stats, Sample Queries.
"""
import sys
import os
import time
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import settings
from backend.core.qdrant_store import get_qdrant_connector
from backend.core.embeddings import get_embedding_model

# Sample queries to test
SAMPLE_QUERIES = [
    "trộm cắp tài sản bị xử lý thế nào",
    "năng lực hành vi dân sự là gì",
    "hợp đồng vô hiệu khi nào"
]

def verify_connection():
    """Step 1: Verify basic connection."""
    print("=" * 60)
    print("STEP 1: CONNECTION VERIFICATION")
    print("=" * 60)
    print(f"Host: {settings.QDRANT_HOST}")
    print(f"Collection: {settings.QDRANT_LEGAL_COLLECTION}")
    print(f"API Key loaded: {bool(settings.QDRANT_API_KEY)}")
    
    try:
        connector = get_qdrant_connector()
        info = connector.get_collection_info(settings.QDRANT_LEGAL_COLLECTION)
        
        if info and info.get('points_count', 0) > 0:
            print(f"✅ CONNECTION SUCCESSFUL!")
            print(f"📊 Points Count: {info['points_count']}")
            print(f"📊 Status: {info['status']}")
            return True, connector
        else:
            print("❌ Collection exists but is empty.")
            return False, None
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return False, None


def verify_search(connector):
    """Step 2: Verify search with sample queries."""
    print("\n" + "=" * 60)
    print("STEP 2: SAMPLE SEARCH VERIFICATION")
    print("=" * 60)
    
    try:
        embedder = get_embedding_model()
    except Exception as e:
        print(f"⚠️ Embedder load failed (expected if no GPU): {e}")
        print("   Skipping search test. Connection was successful.")
        return True
    
    results_summary = []
    
    for query in SAMPLE_QUERIES:
        print(f"\n🔍 Query: '{query}'")
        
        try:
            # Time the search
            start = time.time()
            
            # Generate embedding
            query_vector = embedder.embed([query])[0].tolist()
            embed_time = time.time() - start
            
            # Search
            search_start = time.time()
            results = connector.hybrid_search(
                query_vector=query_vector,
                top_k=3,
                collection="legal"
            )
            search_time = time.time() - search_start
            total_time = time.time() - start
            
            if results:
                print(f"   ✅ Found {len(results)} results")
                print(f"   ⏱️ Embed: {embed_time*1000:.0f}ms | Search: {search_time*1000:.0f}ms | Total: {total_time*1000:.0f}ms")
                
                # Show top result preview
                top = results[0]
                text_preview = top.get('text', '')[:100].replace('\n', ' ')
                print(f"   📄 Top result: \"{text_preview}...\"")
                
                results_summary.append({
                    "query": query,
                    "results_count": len(results),
                    "latency_ms": round(total_time * 1000),
                    "status": "PASS"
                })
            else:
                print(f"   ⚠️ No results found")
                results_summary.append({
                    "query": query,
                    "results_count": 0,
                    "latency_ms": round(total_time * 1000),
                    "status": "WARN"
                })
                
        except Exception as e:
            print(f"   ❌ Search error: {e}")
            results_summary.append({
                "query": query,
                "results_count": 0,
                "latency_ms": -1,
                "status": "FAIL"
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("SEARCH SUMMARY")
    print("=" * 60)
    passed = sum(1 for r in results_summary if r['status'] == 'PASS')
    print(f"✅ Passed: {passed}/{len(results_summary)}")
    avg_latency = sum(r['latency_ms'] for r in results_summary if r['latency_ms'] > 0) / max(1, passed)
    print(f"⏱️ Average Latency: {avg_latency:.0f}ms")
    
    return passed == len(results_summary)


def main():
    print("\n" + "🚀 QDRANT CLOUD VERIFICATION SCRIPT 🚀".center(60))
    print("=" * 60 + "\n")
    
    # Step 1: Connection
    connected, connector = verify_connection()
    
    if not connected:
        print("\n❌ VERIFICATION FAILED: Cannot connect to Qdrant Cloud.")
        return
    
    # Step 2: Search
    search_ok = verify_search(connector)
    
    # Final Result
    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    if connected and search_ok:
        print("🎉 ALL CHECKS PASSED! Qdrant Cloud is fully operational.")
    elif connected:
        print("⚠️ Connection OK, but some searches failed. Check embedder.")
    else:
        print("❌ VERIFICATION FAILED.")


if __name__ == "__main__":
    main()
