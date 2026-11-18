import chromadb

from src.config import AppConfig

config = AppConfig.from_env()

client = chromadb.PersistentClient(
    path=config.chroma_path
)

collection_name = "finance_sample"

try:
        if collection_name in [
            collection.name for collection in client.list_collections()
        ]:
            client.delete_collection(name=collection_name)
            print("Existing collection deleted.")
        else:
            print("Collection does not exist.")
            print(f"Deleted collection: {config.chroma_collection}")
except Exception as exc:
    print(f"Could not delete collection: {exc}")