import faiss
import pickle

from src.utils.encoder import ENCODER


class QueryEngine:
    def __init__(self, index_path, metadata_path, data_type="text"):
        self.encoder = ENCODER.get_encoder()
        self.index = faiss.read_index(index_path)

        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

        self.data_type = data_type

    def query(self, query_input, k=5):
        query_vector = self._encode(query_input)
        distances, indices = self.index.search(query_vector, k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            results.append(self.metadata[idx])

        return results

    def _encode(self, query_input):
        return self.encoder.encode([query_input], convert_to_numpy=True)