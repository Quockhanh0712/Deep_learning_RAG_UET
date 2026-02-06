"""
Qdrant Vector Store Connector with Hybrid Search.
Supports Dense (semantic) + Sparse (BM25) search with RRF fusion.
"""
import os
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

from backend.config import settings
from backend.utils.logger import get_logger
from backend.utils.exceptions import QdrantConnectionError, SearchError

logger = get_logger("qdrant_store")


class QdrantConnector:
    """
    Qdrant Vector Database Connector.
    Provides hybrid search combining dense and sparse vectors.
    """
    
    _instance: Optional["QdrantConnector"] = None
    
    def __new__(cls) -> "QdrantConnector":
        """Singleton pattern to reuse connection."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.host = settings.QDRANT_HOST
        self.port = settings.QDRANT_PORT
        self.api_key = settings.QDRANT_API_KEY
        self.legal_collection = settings.QDRANT_LEGAL_COLLECTION
        self.user_collection = settings.QDRANT_USER_COLLECTION
        
        self._client: Optional[QdrantClient] = None
        self._initialized = True
        logger.info(f"QdrantConnector initialized for {self.host}:{self.port}")
    
    @property
    def client(self) -> QdrantClient:
        """Lazy connection to Qdrant."""
        if self._client is None:
            try:
                # Handle Cloud URL (https://xyz...) vs Host/Port
                if self.host.startswith("http"):
                    # If host is full URL (like Qdrant Cloud), use it directly
                    self._client = QdrantClient(
                        url=self.host,
                        # port=None, # url implies port usually
                        api_key=self.api_key,
                        timeout=30
                    )
                else:
                    self._client = QdrantClient(
                        host=self.host,
                        port=self.port,
                        api_key=self.api_key,
                        timeout=30
                    )
                # Verify connection
                self._client.get_collections()
                logger.info("✓ Connected to Qdrant")
            except Exception as e:
                raise QdrantConnectionError(self.host, self.port, str(e))
        return self._client
    
    def check_connection(self) -> bool:
        """Check if Qdrant is reachable."""
        try:
            collections = self.client.get_collections()
            return len(collections.collections) > 0
        except Exception:
            return False
    
    def list_collections(self) -> List[str]:
        """List all collection names."""
        try:
            collections = self.client.get_collections()
            return [c.name for c in collections.collections]
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            info = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "vectors_count": getattr(info, "vectors_count", info.points_count),
                "points_count": info.points_count,
                "status": getattr(info.status, "value", str(info.status)),
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}
    
    def hybrid_search(
        self,
        query_vector: List[float],
        sparse_vector: Optional[Dict[str, Any]] = None,
        top_k: int = 7,
        collection: str = "legal",
        user_id: Optional[str] = None,
        score_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining dense and sparse vectors.
        Uses Reciprocal Rank Fusion (RRF) for merging results.
        
        Args:
            query_vector: Dense embedding vector (768-dim)
            sparse_vector: Optional sparse (BM25) vector
            top_k: Number of results to return
            collection: 'legal' or 'user'
            user_id: Filter by user_id (for user collection)
            score_threshold: Minimum score threshold
            
        Returns:
            List of search results with metadata
        """
        # Select collection
        collection_name = (
            self.legal_collection if collection == "legal" 
            else self.user_collection
        )
        
        try:
            # Build filter for user collection
            search_filter = None
            if collection == "user" and user_id:
                search_filter = qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="user_id",
                            match=qdrant_models.MatchValue(value=user_id)
                        )
                    ]
                )
            
            # Dense search using query_points() - qdrant-client >= 1.7
            dense_response = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                using="dense",  # Named vector for dense embeddings
                limit=top_k * 2,  # Fetch more for fusion
                query_filter=search_filter,
                score_threshold=score_threshold,
                with_payload=True
            )
            dense_results = dense_response.points
            
            # Debug logging
            if dense_results:
                logger.info(f"[DEBUG] Dense results count: {len(dense_results)}")
                first_payload = dense_results[0].payload if dense_results[0].payload else {}
                logger.info(f"[DEBUG] First result payload keys: {list(first_payload.keys())}")
                logger.info(f"[DEBUG] First result text preview: {str(first_payload.get('text', ''))[:100]}...")
            
            # If no sparse vector, return dense results directly
            if sparse_vector is None:
                return self._format_results(dense_results[:top_k])
            
            # Sparse search (if collection supports it)
            try:
                sparse_response = self.client.query_points(
                    collection_name=collection_name,
                    query=qdrant_models.SparseVector(
                        indices=sparse_vector.get("indices", []),
                        values=sparse_vector.get("values", [])
                    ),
                    using="bm25",
                    limit=top_k * 2,
                    query_filter=search_filter,
                    with_payload=True
                )
                sparse_results = sparse_response.points
            except Exception:
                # Fallback to dense only if sparse fails
                logger.warning("Sparse search failed, using dense only")
                return self._format_results(dense_results[:top_k])
            
            # RRF Fusion
            fused_results = self._rrf_fusion(
                dense_results, 
                sparse_results,
                dense_weight=settings.DENSE_WEIGHT,
                sparse_weight=settings.SPARSE_WEIGHT
            )
            
            return fused_results[:top_k]
            
        except Exception as e:
            logger.error(f"[ERROR] Qdrant search failed: {e}")
            raise SearchError(f"query_vector: {len(query_vector)} dims", str(e))
    
    def _rrf_fusion(
        self,
        dense_results: List,
        sparse_results: List,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF) to merge dense and sparse results.
        
        Formula: score = sum(weight / (k + rank))
        
        Args:
            dense_results: Results from dense search
            sparse_results: Results from sparse search
            dense_weight: Weight for dense scores
            sparse_weight: Weight for sparse scores
            k: RRF constant (typically 60)
            
        Returns:
            Fused and sorted results
        """
        scores: Dict[str, float] = {}
        result_map: Dict[str, Dict[str, Any]] = {}
        
        # Process dense results
        if dense_results:
            logger.info(f"[DEBUG] Dense results sample payload: {dense_results[0].payload}")

        for rank, hit in enumerate(dense_results):
            point_id = str(hit.id)
            rrf_score = dense_weight / (k + rank + 1)
            scores[point_id] = scores.get(point_id, 0) + rrf_score
            
            if point_id not in result_map:
                payload = hit.payload or {}
                # Map payload keys to standard format
                result_map[point_id] = {
                    "id": point_id,
                    "text": payload.get("text") or payload.get("content", ""),
                    "dieu_number": payload.get("dieu_number") or payload.get("dieu"),
                    "khoan_number": payload.get("khoan_number") or payload.get("khoan"),
                    "file_name": payload.get("file_name"),
                    "source_type": payload.get("source_type", "legal"),
                    "dense_score": hit.score,
                    "metadata": payload
                }
        
        # Process sparse results
        for rank, hit in enumerate(sparse_results):
            point_id = str(hit.id)
            rrf_score = sparse_weight / (k + rank + 1)
            scores[point_id] = scores.get(point_id, 0) + rrf_score
            
            if point_id not in result_map:
                payload = hit.payload or {}
                result_map[point_id] = {
                    "id": point_id,
                    "text": payload.get("text") or payload.get("content", ""),
                    "dieu_number": payload.get("dieu") or payload.get("dieu_number"),
                    "khoan_number": payload.get("khoan") or payload.get("khoan_number"),
                    "file_name": payload.get("file_name"),
                    "source_type": payload.get("source_type", "legal"),
                    "sparse_score": hit.score,
                    "metadata": payload
                }
            else:
                result_map[point_id]["sparse_score"] = hit.score
        
        # Sort by fused score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        results = []
        for point_id in sorted_ids:
            result = result_map[point_id]
            result["score"] = scores[point_id]
            results.append(result)
        
        return results
    
    def _format_results(self, results: List) -> List[Dict[str, Any]]:
        """Format Qdrant results to standard format."""
        formatted = []
        for hit in results:
            payload = hit.payload or {}
            formatted.append({
                "id": str(hit.id),
                "text": payload.get("text") or payload.get("content", ""),
                "score": hit.score,
                "dieu_number": payload.get("dieu") or payload.get("dieu_number"),
                "khoan_number": payload.get("khoan") or payload.get("khoan_number"),
                "file_name": payload.get("file_name"),
                "source_type": payload.get("source_type", "legal"),
                "metadata": payload
            })
        return formatted
    
    def insert_points(
        self,
        collection_name: str,
        points: List[Dict[str, Any]]
    ) -> bool:
        """
        Insert vectors into collection.
        
        Args:
            collection_name: Target collection
            points: List of {"id", "vector", "payload"}
            
        Returns:
            Success status
        """
        try:
            qdrant_points = [
                qdrant_models.PointStruct(
                    id=p["id"],
                    vector=p["vector"],
                    payload=p.get("payload", {})
                )
                for p in points
            ]
            
            self.client.upsert(
                collection_name=collection_name,
                points=qdrant_points
            )
            
            logger.info(f"Inserted {len(points)} points to {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert points: {e}")
            return False
    
    def delete_by_filter(
        self,
        collection_name: str,
        filter_conditions: Dict[str, Any]
    ) -> bool:
        """Delete points matching filter conditions."""
        try:
            must_conditions = [
                qdrant_models.FieldCondition(
                    key=key,
                    match=qdrant_models.MatchValue(value=value)
                )
                for key, value in filter_conditions.items()
            ]
            
            self.client.delete(
                collection_name=collection_name,
                points_selector=qdrant_models.FilterSelector(
                    filter=qdrant_models.Filter(must=must_conditions)
                )
            )
            
            logger.info(f"Deleted points from {collection_name} with filter: {filter_conditions}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete points: {e}")
            return False


# Convenience function
@lru_cache()
def get_qdrant_connector() -> QdrantConnector:
    """Get singleton Qdrant connector instance."""
    return QdrantConnector()
