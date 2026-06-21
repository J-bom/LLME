from RAG import file_preprocessor
from RAG import vector_engine as VE
from RAG import vector_store as VS
import torch

class RAG_Engine:
    """
    RAG engine interface, designed for ease of integration in the main program
    """
    def __init__(self, embedding_model):
        """
        initiallizer for the RAG ENGINE
        :param embedding_model: embedding model to use
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.vector_engine = VE.VectorEngine(embedding_model, self.device)

    def search(self, prompt, user_vdb):
        """
        embed prompt and search through the user's personal vector database and return the top_k results
        :param prompt: user prompt
        :param user_vdb: user's vector database objct
        :return: list of relevant context items
        """
        embedded_prompt = self.vector_engine.embed_query(prompt)
        context = user_vdb.search(embedded_prompt,top_k=3)
        context_items = [context_item[0] for context_item in context]

        return context_items

    def initallize_user_vdb(self, user_datapath):
        """
        initiallizes the user's vector database object
        :param user_datapath: data path for user
        :return: the user's vector database object
        """
        return VS.VectorStore(user_datapath, self.device)

    def add_document(self, file_path, user_vdb):
        """
        add document into the user's vector database.
        :param file_path: path to file to add
        :param user_vdb: the user's vector database object
        :return:
        """
        document = file_preprocessor.Document(file_path, 'pdf')
        chunks = file_preprocessor.parse_chunks(document)
        chunks, embeddings_list = self.vector_engine.embed_batch_chunks(chunks)
        user_vdb.add_document_to_store(file_path.name, chunks, embeddings_list)
