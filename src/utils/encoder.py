from sentence_transformers import SentenceTransformer


class Encoder:
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

    def get_encoder(self):
        return self.encoder


ENCODER = Encoder()
