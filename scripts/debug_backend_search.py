import os
import sys
import asyncio
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env
load_dotenv('a:\\NCKH\\.env')

from backend.core.qdrant_store import QdrantConnector
from backend.config import settings

async def main():
    print("Testing QdrantConnector.hybrid_search...")
    print(f"Collection: {settings.QDRANT_LEGAL_COLLECTION}")
    
    try:
        connector = QdrantConnector()
        
        # Dummy query vector
        query_vector = [0.1] * 768
        
        # Test hybrid_search
        results = await connector.hybrid_search(
            query_vector=query_vector,
            collection="user",
            top_k=5,
            score_threshold=0.0
        )
        
        print(f"SUCCESS: hybrid_search returned {len(results)} results")
        for res in results:
            print(f"- {res.get('id')} (score: {res.get('score')})")
            
    except Exception as e:
        print(f"\nERROR in hybrid_search: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
