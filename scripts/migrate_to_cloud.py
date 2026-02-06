import os
import sys
import traceback
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Configuration
SOURCE_URL = "http://localhost:6333"
SOURCE_COLLECTION = "legal_rag_hybrid"
TARGET_URL = "https://f9f847f9-cf96-4fb9-8336-8904be998599.us-east4-0.gcp.cloud.qdrant.io:6333"
TARGET_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.oqAAdUusafc03RTa4JCn46CJJPa6uAsZcn2T51XDbC0"
TARGET_COLLECTION = "legal_rag_vn"
LOG_FILE = "migration_debug.log"

def log(msg):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")

def migrate():
    # Clear log
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("Migration Log Started - Fix v2\n")

    log(f"🚀 Starting migration...")
    
    try:
        source_client = QdrantClient(url=SOURCE_URL)
        target_client = QdrantClient(url=TARGET_URL, api_key=TARGET_API_KEY)
        
        # Get Source Config
        log("🔍 Fetching source info...")
        source_info = source_client.get_collection(SOURCE_COLLECTION)
        config = source_info.config
        
        log(f"   Source Points: {source_info.points_count}")
        
        # Recreate Target
        log(f"🛠 Checking target '{TARGET_COLLECTION}'...")
        if target_client.collection_exists(TARGET_COLLECTION):
           log("   Target exists. Appending data.")
        else:
           log("   Creating target...")
           target_client.recreate_collection(
               collection_name=TARGET_COLLECTION,
               vectors_config=config.params.vectors,
               sparse_vectors_config=config.params.sparse_vectors,
           )
           log("   Target created.")
           
        # Migrate
        log("📦 Transferring data...")
        offset = None
        count = 0
        BATCH = 100
        
        while True:
            points, next_offset = source_client.scroll(
                collection_name=SOURCE_COLLECTION,
                limit=BATCH,
                offset=offset,
                with_payload=True,
                with_vectors=True
            )
            
            if not points:
                break
            
            # CONVERSION STEP: Record -> PointStruct
            points_to_upsert = [
                models.PointStruct(
                    id=point.id,
                    vector=point.vector,
                    payload=point.payload
                )
                for point in points
            ]
                
            target_client.upsert(
                collection_name=TARGET_COLLECTION,
                points=points_to_upsert
            )
            count += len(points)
            if count % 1000 == 0:
                log(f"   Migrated {count}...")
            
            offset = next_offset
            if offset is None:
                break
                
        log(f"✅ Finished. Total: {count}")
        
    except Exception as e:
        log(f"❌ ERROR: {e}")
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            traceback.print_exc(file=f)

if __name__ == "__main__":
    migrate()
