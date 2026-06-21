from sentence_transformers import SentenceTransformer, util
from pathlib import Path
import torch

#number of model chunks to embed at once
BATCH_SIZE = 32

class VectorEngine:
    """
    Vector Engine class, handler of RAG Embeddings
    """
    def __init__(self, model_name='BAAI/bge-base-en-v1.5', device='cpu'):
        """
        initiallizer for the vector engine class, starts up the embedding model
        :param model_name: name of the embedding model
        :param device: device to load model into
        """
        self.embedding_model = SentenceTransformer(model_name_or_path=model_name,device=device)

    def embed_batch_chunks(self, chunks_list):
        """
        embed batches of chunks for fast document processing
        :param chunks_list: list of chunks
        :return: list of chunks + embeddings
        """
        text_contents = [chunk_data['chunk'].contents for chunk_data in chunks_list]
        batch_embeddings = self.embedding_model.encode(text_contents, batch_size=BATCH_SIZE, convert_to_tensor=True)
        for i, chunk_data in enumerate(chunks_list):
            chunk_data['chunk'].embedding = batch_embeddings[i].tolist()
        return chunks_list, batch_embeddings

    def embed_query(self, query):
        """
        embed user prompt
        :param query: user prompt
        :return: embedded user prompt
        """
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        return query_embedding


