from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")


class Encoder:
    """A class to manage text encoding using a SentenceTransformer model.

    The `Encoder` class initializes a SentenceTransformer model (default: 'all-MiniLM-L6-v2')
    and provides access to it for encoding text into dense vector representations, suitable
    for tasks like similarity search or clustering.

    Attributes:
        encoder (SentenceTransformer): The SentenceTransformer model used for encoding.
    """

    def __init__(self):
        """Initialize the Encoder with a SentenceTransformer model.

        Loads the 'all-MiniLM-L6-v2' SentenceTransformer model, which is a lightweight
        model optimized for generating sentence embeddings.

        """
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

    def get_encoder(self) -> SentenceTransformer:
        """Retrieve the SentenceTransformer model instance.

        Returns:
            SentenceTransformer: The initialized SentenceTransformer model.

        Example:
            encoder = Encoder()
            model = encoder.get_encoder()
            embeddings = model.encode(["This is a sentence."])
        """
        return self.encoder


# Singleton instance of the Encoder class for global access
ENCODER = Encoder()
