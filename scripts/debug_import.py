try:
    print("Importing qdrant_client...")
    import qdrant_client
    print("Success qdrant_client")
    print("Importing models...")
    from qdrant_client.http import models
    print("Success models")
except Exception as e:
    print(f"Error: {e}")
