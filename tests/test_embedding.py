from src.strategies.embeddings import SentenceTransformerEmbedder

embedder = SentenceTransformerEmbedder()

texts = [
    "The approved project budget was 500000.",
    "The financial report contains revenue details.",
    "The loan interest rate is 8 percent."
]

# Batch embedding
embeddings = embedder.embed_texts(texts)

print("Document embeddings shape:", embeddings.shape)
print("Embedding dimension:", embedder.get_dimensions())

# Query embedding
query_embedding = embedder.embed_query(
    "What was the approved project budget?"
)

print("Query embedding shape:", query_embedding.shape)

# Clone interface
encoded = embedder.encode(texts)
print("Clone interface vector count:", len(encoded))