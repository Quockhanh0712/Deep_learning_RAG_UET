"""
Search API Endpoint - Hybrid Search.
"""
from fastapi import APIRouter, HTTPException
from typing import Optional

from backend.db.schemas import SearchRequest, SearchResponse
from backend.core.rag_pipeline import get_rag_pipeline
from backend.utils.logger import get_logger
from backend.utils.exceptions import SearchError

logger = get_logger("api.search")

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Perform hybrid search on legal documents.
    
    Combines dense (semantic) and sparse (BM25) search with RRF fusion.
    Optionally uses Vietnamese reranker for improved accuracy.
    
    Args:
        request: SearchRequest with query, top_k, search_mode, etc.
        
    Returns:
        SearchResponse with results and metadata
    """
    logger.info(f"Search request: query='{request.query[:50]}...', mode={request.search_mode}")
    
    try:
        pipeline = get_rag_pipeline()
        
        result = await pipeline.search_only(
            query=request.query,
            user_id=request.user_id,
            search_mode=request.search_mode,
            top_k=request.top_k,
            reranker_enabled=request.reranker_enabled
        )
        
        logger.info(f"Search completed: {result['total']} results in {result['search_time_ms']:.0f}ms")
        
        return SearchResponse(
            results=result["results"],
            total=result["total"],
            query=result["query"],
            search_mode=result["search_mode"]
        )
        
    except SearchError as e:
        logger.error(f"Search error: {e.message}")
        raise HTTPException(status_code=500, detail=e.to_dict())
    except Exception as e:
        logger.exception(f"Unexpected search error: {e}")
        raise HTTPException(status_code=500, detail={"error": "SearchFailed", "message": str(e)})


@router.get("/search/collections")
async def list_collections():
    """List available search collections."""
    try:
        from backend.core.qdrant_store import get_qdrant_connector
        connector = get_qdrant_connector()
        
        collections = []
        for name in connector.list_collections():
            info = connector.get_collection_info(name)
            if info:
                collections.append(info)
        
        return {"collections": collections}
        
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})
