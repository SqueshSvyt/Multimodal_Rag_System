import faiss
import pickle
import numpy as np
from src.utils.encoder import ENCODER


class QueryEngine:
    """A class to perform similarity search queries using a FAISS index and an encoder.

    The `QueryEngine` loads a pre-built FAISS index and associated metadata, allowing
    users to query the index with text or other data types and retrieve the most similar
    items based on vector similarity. It relies on an encoder to transform queries into
    vector representations compatible with the FAISS index.

    Attributes:
        encoder: The encoder instance used to convert queries into vectors.
        index: The FAISS index containing pre-computed vector representations.
        metadata: A list or dictionary containing metadata for indexed items.
        data_type (str): The type of data being queried (default: "text").
    """

    def __init__(self, index_path: str, metadata_path: str, data_type: str = "text"):
        """Initialize the QueryEngine with a FAISS index and metadata.

        Args:
            index_path (str): Path to the pre-built FAISS index file.
            metadata_path (str): Path to the pickled metadata file.
            data_type (str, optional): Type of data for queries (e.g., "text"). Defaults to "text".
        """
        self.encoder = ENCODER.get_encoder()
        self.reranker = ENCODER.get_reranker()

        self.index = faiss.read_index(index_path)

        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

        self.data_type = data_type

    def query(self, query_input: str, k: int = 20, rerank_k: int = 20) -> list:
        """Perform a similarity search to retrieve the top-k most similar items.

        Encodes the query input into a vector and searches the FAISS index to find
        the `k` nearest neighbors. Returns the metadata associated with the matching
        indices.

        Args:
            query_input (str): The input query (e.g., text string) to search for.
            k (int, optional): Number of nearest neighbors to retrieve. Defaults to 5.

        Returns:
            list: A list of metadata entries corresponding to the top-k similar items.
        """
        query_vector = self._encode(query_input)

        distances, indices = self.index.search(query_vector, k)

        docs = []
        for idx, dist in zip(indices[0], distances[0]):
            docs.append(self.metadata[idx])

        reranked = self._rerank(query_input, docs)
        return reranked[:rerank_k]

    def _encode(self, query_input: str) -> np.ndarray:
        """Encode the query input into a vector representation.

        Uses the configured encoder to transform the query input into a numpy array
        suitable for FAISS search.

        Args:
            query_input (str): The input query to encode.

        Returns:
            np.ndarray: The encoded query vector as a numpy array.
        """
        return self.encoder.encode([query_input], convert_to_numpy=True)

    def _rerank(self, query: str, documents: list[dict]) -> list[dict]:
        """
        Rerank retrieved documents using a cross-encoder reranker.

        Args:
            query (str): The input query.
            documents (list[dict]): A list of metadata entries, each containing at least a 'text' field.

        Returns:
            list[dict]: The input documents sorted by their relevance to the query.
        """
        if self.data_type == 'image':
            pairs = [(query, doc["image_descriptions"]) for doc in documents]
        else:
            pairs = [(query, doc["content"]) for doc in documents]

        scores = self.reranker.predict(pairs)

        reranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in reranked]
