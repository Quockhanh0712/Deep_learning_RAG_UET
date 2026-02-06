"""
Integration Tests for API Endpoints.
Tests actual HTTP requests to the API with real services.
"""
import sys
from pathlib import Path
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ==================== Status API Tests ====================

@pytest.mark.integration
@pytest.mark.asyncio
class TestStatusAPI:
    """Test /api/status endpoint."""
    
    async def test_root_endpoint(self, async_client):
        """Root endpoint should return API info."""
        response = await async_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
    
    async def test_health_endpoint(self, async_client):
        """Health endpoint should return ok."""
        response = await async_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    async def test_quick_status(self, async_client):
        """Quick status should return timestamp."""
        response = await async_client.get("/api/status/quick")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "timestamp" in data
        assert "gpu" in data
    
    async def test_full_status(self, async_client):
        """Full status should check all components."""
        response = await async_client.get("/api/status")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check all components are present
        assert "qdrant" in data
        assert "ollama" in data
        assert "embedding" in data
        assert "database" in data
        assert "gpu_available" in data


# ==================== Search API Tests ====================

@pytest.mark.integration
@pytest.mark.asyncio
class TestSearchAPI:
    """Test /api/search endpoint."""
    
    async def test_search_basic(self, async_client, sample_query):
        """Basic search should return results."""
        response = await async_client.post("/api/search", json={
            "query": sample_query,
            "top_k": 5,
            "search_mode": "legal",
            "reranker_enabled": False
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "results" in data
        assert "total" in data
        assert "query" in data
        assert data["query"] == sample_query
    
    async def test_search_with_reranker(self, async_client, sample_query):
        """Search with reranker should work."""
        response = await async_client.post("/api/search", json={
            "query": sample_query,
            "top_k": 3,
            "search_mode": "legal",
            "reranker_enabled": True
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "results" in data
        # Results should have scores
        if data["results"]:
            assert "score" in data["results"][0]
    
    async def test_search_empty_query(self, async_client):
        """Empty query should return validation error."""
        response = await async_client.post("/api/search", json={
            "query": "",
            "top_k": 5
        })
        
        # FastAPI validation should reject empty query
        assert response.status_code == 422
    
    async def test_list_collections(self, async_client):
        """Should list available collections."""
        response = await async_client.get("/api/search/collections")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "collections" in data
        assert isinstance(data["collections"], list)


# ==================== Chat API Tests ====================

@pytest.mark.integration
@pytest.mark.asyncio
class TestChatAPI:
    """Test /api/chat endpoint."""
    
    async def test_chat_basic(self, async_client, sample_query, sample_user_id):
        """Basic chat should return answer."""
        response = await async_client.post("/api/chat", json={
            "message": sample_query,
            "user_id": sample_user_id,
            "search_mode": "legal",
            "reranker_enabled": False
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert "message_id" in data
        assert len(data["answer"]) > 0
    
    async def test_chat_with_session(self, async_client, sample_query, sample_user_id, sample_session_id):
        """Chat with session ID should reuse session."""
        # First message
        response1 = await async_client.post("/api/chat", json={
            "message": sample_query,
            "user_id": sample_user_id,
            "session_id": sample_session_id,
            "search_mode": "legal"
        })
        
        assert response1.status_code == 200
        data1 = response1.json()
        
        # Second message in same session
        response2 = await async_client.post("/api/chat", json={
            "message": "Còn hình phạt bổ sung thì sao?",
            "user_id": sample_user_id,
            "session_id": sample_session_id,
            "search_mode": "legal"
        })
        
        assert response2.status_code == 200
        data2 = response2.json()
        
        # Should be same session
        assert data1["session_id"] == data2["session_id"]
    
    async def test_chat_returns_sources(self, async_client, sample_query, sample_user_id):
        """Chat should return source citations."""
        response = await async_client.post("/api/chat", json={
            "message": sample_query,
            "user_id": sample_user_id,
            "search_mode": "legal"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have sources
        assert "sources" in data
        assert isinstance(data["sources"], list)
        
        # If sources present, check structure
        if data["sources"]:
            source = data["sources"][0]
            assert "text" in source
            assert "score" in source
    
    async def test_list_sessions(self, async_client, sample_user_id):
        """Should list user sessions."""
        response = await async_client.get(f"/api/chat/sessions/{sample_user_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "sessions" in data
        assert "total" in data


# ==================== Document API Tests ====================

@pytest.mark.integration
@pytest.mark.asyncio
class TestDocumentAPI:
    """Test /api/documents and /api/upload endpoints."""
    
    async def test_list_documents(self, async_client, sample_user_id):
        """Should list user documents."""
        response = await async_client.get(f"/api/documents/{sample_user_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "documents" in data
        assert "total" in data


# ==================== Error Handling Tests ====================

@pytest.mark.integration
@pytest.mark.asyncio
class TestErrorHandling:
    """Test API error handling."""
    
    async def test_invalid_json(self, async_client):
        """Invalid JSON should return 422."""
        response = await async_client.post(
            "/api/search",
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    async def test_missing_required_field(self, async_client):
        """Missing required field should return 422."""
        response = await async_client.post("/api/chat", json={
            "message": "test"
            # Missing user_id
        })
        
        assert response.status_code == 422
    
    async def test_404_not_found(self, async_client):
        """Invalid endpoint should return 404."""
        response = await async_client.get("/api/nonexistent")
        
        assert response.status_code == 404


# ==================== Run Tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
