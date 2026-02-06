"""
Test Hybrid Legal Search

Kiểm tra tìm kiếm kết hợp Graph + Vector
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables BEFORE importing modules
from dotenv import load_dotenv
load_dotenv()

from src.hybrid_legal_search import HybridLegalSearch
from src.graph_db import LegalGraphDB


def test_graph_search():
    """Test tìm kiếm trên Graph Database"""
    print("\n" + "="*60)
    print("TEST 1: Graph Database Search")
    print("="*60)
    
    graph = LegalGraphDB()
    # Graph is auto-loaded in __init__ if persist_path exists
    
    stats = graph.stats()
    print(f"✓ Graph loaded: {stats['total_nodes']} nodes")
    print(f"  Node types: {stats['node_types']}")
    
    # Test search
    results = graph.search_by_content("kết hôn", limit=5)
    print(f"\n🔍 Search 'kết hôn': {len(results)} results")
    for node_id, data in results[:3]:
        content = data.get('content', '')[:80]
        print(f"  - {node_id[:50]}: {content}...")
    
    print("\n✅ Graph search passed!")
    return True


def test_hybrid_search():
    """Test Hybrid Search (Graph + Vector)"""
    print("\n" + "="*60)
    print("TEST 2: Hybrid Search (Graph + Vector)")
    print("="*60)
    
    try:
        hybrid = HybridLegalSearch()
        print("✓ HybridLegalSearch initialized")
        
        # Test vector-only search
        results = hybrid.search(
            query="điều kiện kết hôn theo pháp luật",
            strategy='vector_only',
            k=5
        )
        print(f"\n🔍 Vector-only search: {len(results)} results")
        
        if len(results) == 0:
            print("❌ FAILED: No results returned from vector search")
            return False
            
        for r in results[:3]:
            print(f"  - Score: {r.score:.3f} | {r.content[:60]}...")
        
        # Test graph-enhanced search
        results = hybrid.search(
            query="điều kiện kết hôn theo pháp luật",
            strategy='graph_enhanced',
            k=5
        )
        print(f"\n🔍 Graph-enhanced search: {len(results)} results")
        
        if len(results) == 0:
            print("❌ FAILED: No results from graph-enhanced search")
            return False
            
        for r in results[:3]:
            print(f"  - Score: {r.score:.3f} | Full context: {len(r.full_context) if r.full_context else 0} chars")
        
        print("\n✅ Hybrid search passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Hybrid search error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rag_context():
    """Test RAG context building"""
    print("\n" + "="*60)
    print("TEST 3: RAG Context Building")
    print("="*60)
    
    try:
        hybrid = HybridLegalSearch()
        
        # First search, then build context
        results = hybrid.search(
            query="thủ tục đăng ký kết hôn",
            strategy='graph_enhanced',
            k=5
        )
        
        if len(results) == 0:
            print("❌ FAILED: No results to build context from")
            return False
        
        context = hybrid.build_rag_context(results=results)
        
        if len(context) == 0:
            print("❌ FAILED: Built context is empty")
            return False
        
        print(f"✓ Built RAG context: {len(context)} chars")
        print(f"\nSample context:\n{context[:500]}...")
        
        print("\n✅ RAG context passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ RAG context error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_search_by_law():
    """Test search within specific law"""
    print("\n" + "="*60)
    print("TEST 4: Search by Law")
    print("="*60)
    
    graph = LegalGraphDB()
    # Graph is auto-loaded in __init__
    
    # Get some law IDs (extract law_id from node data, not node_id)
    law_ids = []
    for n, d in graph.graph.nodes(data=True):
        if d.get('type') == 'law':
            law_ids.append(d.get('law_id', n))
            if len(law_ids) >= 5:
                break
    
    print(f"Sample law_ids: {law_ids[:3]}")
    
    if law_ids:
        results = graph.search_by_law(law_ids[0])
        print(f"\n🔍 Nodes in '{law_ids[0][:40]}...': {len(results)}")
        
        if len(results) == 0:
            print("❌ FAILED: No nodes found for law")
            return False
            
        for node_id, data in results[:3]:
            article_title = data.get('article_title', 'N/A')
            content = data.get('content', '')[:50]
            print(f"  - {article_title}: {content}...")
    
    print("\n✅ Search by law passed!")
    return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🧪 HYBRID LEGAL SEARCH TESTS")
    print("="*60)
    
    results = []
    
    # Test 1: Graph search
    results.append(("Graph Search", test_graph_search()))
    
    # Test 2: Hybrid search
    results.append(("Hybrid Search", test_hybrid_search()))
    
    # Test 3: RAG context
    results.append(("RAG Context", test_rag_context()))
    
    # Test 4: Search by law
    results.append(("Search by Law", test_search_by_law()))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
