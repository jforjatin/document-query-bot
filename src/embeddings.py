"""
    Embeddings Module.

    This module provides utilities for generating text embeddings using SentenceTransformers
    and querying a vector database for similarity results.
"""
from sentence_transformers import SentenceTransformer

# Load the model once for reuse
model = SentenceTransformer("model/all-MiniLM-L6-v2")

def create_embeddings(text_chunks):
    """
        Generate embeddings for the given list of text chunks.
    """
    if not text_chunks:
        raise ValueError("Text chunks list is empty. Cannot create embeddings.")
    try:
        embeddings = model.encode(text_chunks)
        return list(zip(embeddings, text_chunks))
    except Exception as e:
        raise RuntimeError(f"Error generating embeddings: {str(e)}")


def query_embeddings(query, db_path, top_k=5):
    """
        Query the vector database for the most similar results.
    """
    from src.vector_db import load_vector_db
    vector_db = load_vector_db(db_path)
    query_embedding = model.encode([query])[0]
    results = vector_db.search(query_embedding, top_k=top_k)

    # Calculate cosine similarity for ranking results
    ranked_results = []
    for text, distance in results:
        similarity = 1 - (distance / 2)  # FAISS distance is squared, so convert to similarity (0-1 scale)
        ranked_results.append((text, similarity))

    return sorted(ranked_results, key=lambda x: x[1], reverse=True)
