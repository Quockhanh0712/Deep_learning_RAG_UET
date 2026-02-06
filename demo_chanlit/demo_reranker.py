#!/usr/bin/env python3
"""
Demo: Cross-Encoder Reranking với BGE-Reranker-v2-M3

Minh họa cách sử dụng Cross-Encoder để re-rank kết quả retrieval,
cải thiện độ chính xác so với chỉ dùng vector search.

Model: BAAI/bge-reranker-v2-m3
- Multilingual support (bao gồm Vietnamese)
- ~570MB model size
- GPU accelerated

Usage:
    python demo_reranker.py
    python demo_reranker.py --query "Tội tham nhũng bị phạt bao nhiêu năm?"
    python demo_reranker.py --interactive
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

# Add project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_TF"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Cross-Encoder Reranker Class
# ==============================================================================

@dataclass
class RerankedDocument:
    """Document with reranking scores"""
    content: str
    original_score: float
    rerank_score: float
    final_score: float
    rank_before: int
    rank_after: int
    metadata: Dict = None


class BGEReranker:
    """
    Cross-Encoder Reranker using BAAI/bge-reranker-v2-m3
    
    Cross-Encoder vs Bi-Encoder:
    - Bi-Encoder: Encode query và document separately → Fast but less accurate
    - Cross-Encoder: Encode (query, document) pair together → Slower but more accurate
    
    Use Cross-Encoder to rerank top-K results from initial retrieval.
    """
    
    SUPPORTED_MODELS = [
        "BAAI/bge-reranker-v2-m3",      # Multilingual, best for Vietnamese
        "BAAI/bge-reranker-large",       # English focused, larger
        "BAAI/bge-reranker-base",        # English focused, smaller
        "itdainb/PhoRanker",             # Vietnamese specific
        "cross-encoder/ms-marco-MiniLM-L-6-v2",  # Fast English
    ]
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        use_gpu: bool = True,
        max_length: int = 512,
        batch_size: int = 16
    ):
        """
        Initialize BGE Reranker
        
        Args:
            model_name: HuggingFace model name
            use_gpu: Use CUDA if available
            max_length: Max sequence length
            batch_size: Batch size for scoring
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.max_length = max_length
        self.batch_size = batch_size
        
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self._loaded = False
        
    def load(self):
        """Load reranker model"""
        if self._loaded:
            return
        
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        logger.info(f"[RERANKER] Loading {self.model_name}...")
        start = time.time()
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        
        # Move to GPU if available
        if self.use_gpu:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
                self.model.to(self.device)
                logger.info(f"[RERANKER] Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                logger.warning("[RERANKER] CUDA not available, using CPU")
        
        self.model.eval()
        self._loaded = True
        
        elapsed = time.time() - start
        logger.info(f"[RERANKER] Loaded on {self.device} in {elapsed:.2f}s")
    
    def compute_scores(
        self,
        query: str,
        documents: List[str]
    ) -> List[float]:
        """
        Compute relevance scores for query-document pairs
        
        Args:
            query: Search query
            documents: List of document texts
            
        Returns:
            List of relevance scores (0-1, higher = more relevant)
        """
        if not self._loaded:
            self.load()
        
        if not documents:
            return []
        
        import torch
        
        scores = []
        
        # Process in batches
        for i in range(0, len(documents), self.batch_size):
            batch_docs = documents[i:i + self.batch_size]
            
            # Create (query, document) pairs
            # Cross-encoder processes pairs together for better understanding
            pairs = [[query, doc[:2000]] for doc in batch_docs]  # Truncate long docs
            
            # Tokenize pairs
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            # Compute scores
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Get logits
                if hasattr(outputs, 'logits'):
                    batch_scores = outputs.logits.squeeze(-1)
                else:
                    batch_scores = outputs[0].squeeze(-1)
                
                # Normalize to 0-1 using sigmoid
                batch_scores = torch.sigmoid(batch_scores)
                scores.extend(batch_scores.cpu().tolist())
        
        return scores
    
    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = None,
        content_key: str = "content",
        score_key: str = "score",
        alpha: float = 0.5
    ) -> List[RerankedDocument]:
        """
        Rerank documents using cross-encoder scores
        
        Combines original retrieval score with reranker score:
        final_score = alpha * rerank_score + (1 - alpha) * original_score
        
        Args:
            query: Search query
            documents: List of document dicts
            top_k: Number of results to return (None = all)
            content_key: Key for document content
            score_key: Key for original score
            alpha: Weight for reranker score (0=original only, 1=rerank only)
            
        Returns:
            List of RerankedDocument sorted by final_score
        """
        if not documents:
            return []
        
        start = time.time()
        
        # Extract contents and scores
        contents = [doc.get(content_key, "") for doc in documents]
        original_scores = [doc.get(score_key, 0.0) for doc in documents]
        
        # Compute reranker scores
        rerank_scores = self.compute_scores(query, contents)
        
        # Combine scores and create results
        results = []
        for i, (doc, orig_score, rerank_score) in enumerate(zip(documents, original_scores, rerank_scores)):
            final_score = alpha * rerank_score + (1 - alpha) * orig_score
            
            results.append(RerankedDocument(
                content=contents[i],
                original_score=orig_score,
                rerank_score=rerank_score,
                final_score=final_score,
                rank_before=i + 1,
                rank_after=0,  # Will be set after sorting
                metadata=doc.get("metadata", {})
            ))
        
        # Sort by final score (descending)
        results.sort(key=lambda x: x.final_score, reverse=True)
        
        # Update ranks after sorting
        for i, result in enumerate(results):
            result.rank_after = i + 1
        
        elapsed = time.time() - start
        logger.info(f"[RERANKER] Reranked {len(documents)} docs in {elapsed:.2f}s")
        
        # Return top_k if specified
        if top_k:
            results = results[:top_k]
        
        return results


# ==============================================================================
# Demo Functions
# ==============================================================================

def create_sample_documents() -> List[Dict]:
    """Create sample legal documents for demo"""
    return [
        {
            "content": """Điều 353. Tội tham ô tài sản
1. Người nào lợi dụng chức vụ, quyền hạn chiếm đoạt tài sản mà mình có trách nhiệm quản lý trị giá từ 2.000.000 đồng đến dưới 100.000.000 đồng hoặc dưới 2.000.000 đồng nhưng thuộc một trong các trường hợp sau đây, thì bị phạt tù từ 02 năm đến 07 năm.""",
            "score": 0.85,
            "metadata": {"source": "Bộ luật Hình sự 2015", "article": "353"}
        },
        {
            "content": """Điều 354. Tội nhận hối lộ
1. Người nào lợi dụng chức vụ, quyền hạn trực tiếp hoặc qua trung gian nhận hoặc sẽ nhận tiền, tài sản hoặc lợi ích vật chất khác trị giá từ 2.000.000 đồng đến dưới 100.000.000 đồng, thì bị phạt tù từ 02 năm đến 07 năm.""",
            "score": 0.82,
            "metadata": {"source": "Bộ luật Hình sự 2015", "article": "354"}
        },
        {
            "content": """Điều 355. Tội lạm dụng chức vụ, quyền hạn chiếm đoạt tài sản
1. Người nào lạm dụng chức vụ, quyền hạn chiếm đoạt tài sản của người khác trị giá từ 2.000.000 đồng đến dưới 100.000.000 đồng, thì bị phạt tù từ 01 năm đến 06 năm.""",
            "score": 0.80,
            "metadata": {"source": "Bộ luật Hình sự 2015", "article": "355"}
        },
        {
            "content": """Điều 356. Tội lợi dụng chức vụ, quyền hạn trong khi thi hành công vụ
Người nào vì vụ lợi hoặc động cơ cá nhân khác mà lợi dụng chức vụ, quyền hạn làm trái công vụ gây thiệt hại về tài sản từ 10.000.000 đồng đến dưới 200.000.000 đồng, thì bị phạt tù từ 01 năm đến 05 năm.""",
            "score": 0.78,
            "metadata": {"source": "Bộ luật Hình sự 2015", "article": "356"}
        },
        {
            "content": """Điều 123. Tội giết người
1. Người nào giết người thuộc một trong các trường hợp sau đây, thì bị phạt tù từ 12 năm đến 20 năm, tù chung thân hoặc tử hình.""",
            "score": 0.65,
            "metadata": {"source": "Bộ luật Hình sự 2015", "article": "123"}
        },
        {
            "content": """Điều 168. Tội cướp tài sản
1. Người nào dùng vũ lực, đe dọa dùng vũ lực ngay tức khắc hoặc có hành vi khác làm cho người bị tấn công lâm vào tình trạng không thể chống cự được nhằm chiếm đoạt tài sản, thì bị phạt tù từ 03 năm đến 10 năm.""",
            "score": 0.60,
            "metadata": {"source": "Bộ luật Hình sự 2015", "article": "168"}
        },
        {
            "content": """Điều 357. Tội lạm quyền trong khi thi hành công vụ
Người nào vì vụ lợi hoặc động cơ cá nhân khác mà vượt quá quyền hạn của mình làm trái công vụ gây thiệt hại cho lợi ích của Nhà nước, quyền, lợi ích hợp pháp của tổ chức, cá nhân, thì bị phạt tù từ 01 năm đến 07 năm.""",
            "score": 0.75,
            "metadata": {"source": "Bộ luật Hình sự 2015", "article": "357"}
        },
        {
            "content": """Điều 358. Tội lợi dụng ảnh hưởng đối với người có chức vụ quyền hạn để trục lợi
Người nào lợi dụng ảnh hưởng đối với người có chức vụ, quyền hạn nhận tiền, tài sản hoặc lợi ích vật chất khác trị giá từ 2.000.000 đồng đến dưới 100.000.000 đồng, thì bị phạt tù từ 01 năm đến 06 năm.""",
            "score": 0.72,
            "metadata": {"source": "Bộ luật Hình sự 2015", "article": "358"}
        },
    ]


def print_comparison(
    query: str,
    original_docs: List[Dict],
    reranked_results: List[RerankedDocument]
):
    """Print before/after comparison"""
    
    print("\n" + "=" * 80)
    print(f"📝 QUERY: {query}")
    print("=" * 80)
    
    # Before reranking
    print("\n📋 BEFORE RERANKING (Original Retrieval Order):")
    print("-" * 80)
    for i, doc in enumerate(original_docs[:5], 1):
        content_preview = doc["content"][:100].replace("\n", " ") + "..."
        print(f"  #{i} [Score: {doc['score']:.3f}] {content_preview}")
    
    # After reranking
    print("\n✨ AFTER RERANKING (Cross-Encoder Reordered):")
    print("-" * 80)
    for result in reranked_results[:5]:
        content_preview = result.content[:100].replace("\n", " ") + "..."
        rank_change = result.rank_before - result.rank_after
        if rank_change > 0:
            change_str = f"↑{rank_change}"
        elif rank_change < 0:
            change_str = f"↓{abs(rank_change)}"
        else:
            change_str = "="
        
        print(f"  #{result.rank_after} [{change_str}] [Orig: {result.original_score:.3f}] [Rerank: {result.rerank_score:.3f}] [Final: {result.final_score:.3f}]")
        print(f"      {content_preview}")
    
    print("\n" + "=" * 80)


def demo_basic(query: str = None):
    """Basic demo with sample documents"""
    
    print("\n" + "🔄" * 40)
    print("       DEMO: BGE-Reranker-v2-M3 Cross-Encoder")
    print("🔄" * 40)
    
    # Default query
    if not query:
        query = "Tội tham nhũng bị phạt bao nhiêu năm tù?"
    
    # Create sample documents
    documents = create_sample_documents()
    
    # Initialize reranker
    print("\n📥 Loading Cross-Encoder model...")
    reranker = BGEReranker(
        model_name="BAAI/bge-reranker-v2-m3",
        use_gpu=True
    )
    reranker.load()
    
    # Rerank documents
    print("\n🔄 Reranking documents...")
    results = reranker.rerank(
        query=query,
        documents=documents,
        top_k=5,
        alpha=0.7  # 70% weight to reranker score
    )
    
    # Print comparison
    print_comparison(query, documents, results)
    
    # Print insights
    print("\n💡 INSIGHTS:")
    print("-" * 80)
    print(f"  • Cross-Encoder model: {reranker.model_name}")
    print(f"  • Device: {reranker.device}")
    print(f"  • Documents reranked: {len(documents)}")
    print(f"  • Top result changed: {'Yes' if results[0].rank_before != 1 else 'No'}")
    
    # Show rank changes
    improvements = sum(1 for r in results if r.rank_before > r.rank_after)
    print(f"  • Documents moved up: {improvements}")
    
    return results


def demo_with_rag_pipeline(query: str = None):
    """Demo with real RAG pipeline (requires Qdrant running)"""
    
    print("\n" + "🔄" * 40)
    print("    DEMO: Reranker with Legal RAG Pipeline")
    print("🔄" * 40)
    
    if not query:
        query = "Mức phạt tội nhận hối lộ là bao nhiêu?"
    
    try:
        from src.legal_rag_pipeline import get_legal_rag_pipeline
        from src.reranker import LegalReranker
        
        # Initialize pipeline
        print("\n📥 Initializing RAG Pipeline...")
        pipeline = get_legal_rag_pipeline()
        
        # Query without reranker
        print("\n🔍 Querying WITHOUT reranker...")
        result_no_rerank = pipeline.query(
            question=query,
            top_k=10,
            use_reranker=False
        )
        
        # Query with reranker
        print("\n🔍 Querying WITH reranker...")
        result_with_rerank = pipeline.query(
            question=query,
            top_k=10,
            use_reranker=True
        )
        
        # Compare results
        print("\n" + "=" * 80)
        print(f"📝 QUERY: {query}")
        print("=" * 80)
        
        print("\n📋 WITHOUT Reranker:")
        print(f"  • Retrieval time: {result_no_rerank.retrieval_time:.2f}s")
        print(f"  • Generation time: {result_no_rerank.generation_time:.2f}s")
        print(f"  • Citations: {len(result_no_rerank.citations)}")
        
        print("\n✨ WITH Reranker:")
        print(f"  • Retrieval time: {result_with_rerank.retrieval_time:.2f}s")
        print(f"  • Rerank time: {result_with_rerank.rerank_time:.2f}s")
        print(f"  • Generation time: {result_with_rerank.generation_time:.2f}s")
        print(f"  • Citations: {len(result_with_rerank.citations)}")
        
        print("\n📝 Answer (with reranker):")
        print("-" * 80)
        print(result_with_rerank.answer)
        
    except Exception as e:
        logger.error(f"Failed to run RAG demo: {e}")
        print(f"\n❌ Error: {e}")
        print("💡 Make sure Qdrant is running: docker start qdrant")


def demo_interactive():
    """Interactive demo mode"""
    
    print("\n" + "🔄" * 40)
    print("       INTERACTIVE RERANKER DEMO")
    print("🔄" * 40)
    
    # Initialize reranker
    print("\n📥 Loading Cross-Encoder model...")
    reranker = BGEReranker(
        model_name="BAAI/bge-reranker-v2-m3",
        use_gpu=True
    )
    reranker.load()
    
    documents = create_sample_documents()
    
    print("\n✅ Ready! Enter your legal questions (type 'quit' to exit)")
    print("-" * 80)
    
    while True:
        try:
            query = input("\n❓ Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not query:
                continue
            
            # Rerank
            results = reranker.rerank(
                query=query,
                documents=documents,
                top_k=5,
                alpha=0.7
            )
            
            # Show results
            print_comparison(query, documents, results)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Demo: Cross-Encoder Reranking with BGE-Reranker-v2-M3"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Custom query for demo"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run interactive mode"
    )
    parser.add_argument(
        "--with-rag",
        action="store_true",
        help="Demo with real RAG pipeline (requires Qdrant)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="BAAI/bge-reranker-v2-m3",
        help="Reranker model name"
    )
    
    args = parser.parse_args()
    
    if args.interactive:
        demo_interactive()
    elif args.with_rag:
        demo_with_rag_pipeline(args.query)
    else:
        demo_basic(args.query)


if __name__ == "__main__":
    main()
