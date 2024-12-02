"""
    Vector Database Module.

    This module implements a vector database using FAISS for
    storing and searching embeddings.
"""
import pickle
import faiss
import numpy as np

class VectorDB:
    """
        Vector Database for storing and querying embeddings.
    """
    def __init__(self):
        self.index = faiss.IndexFlatL2(384)  # Dimension of MiniLM embeddings
        self.texts = []

    def add(self, embeddings):
        """
            Adds embeddings to the database.
        """
        vectors, texts = zip(*embeddings)
        self.index.add(np.array(vectors))
        self.texts.extend(texts)

    def search(self, query_vector, top_k=5):
        """
            Searches the database for the most similar embeddings.
        """
        distances, indices = self.index.search(np.array([query_vector]), k=top_k)
        return [(self.texts[i], distances[0][idx]) for idx, i in enumerate(indices[0])]

def save_vector_db(db, path):
    """
        Saves the vector database to a file.
    """
    with open(path, "wb") as f:
        pickle.dump(db, f)

def load_vector_db(path):
    """
        Loads the vector database from a file.
    """
    try:
        with open(path, "rb") as f:
            db = pickle.load(f)
            if not isinstance(db, VectorDB):
                raise ValueError("Loaded file is not a valid VectorDB instance.")
            return db
    except FileNotFoundError:
        return VectorDB()
    except Exception as e:
        raise RuntimeError(f"Error loading vector database: {str(e)}")
