import torch
import sqlite3
import json
from pathlib import Path
from sentence_transformers import util


class VectorStore:
    """
    manages access and control of the RAG context database and vector storage.
    """
    def __init__(self, user_data_path, device='cpu'):
        """
        initiallizer for the vector store class, initiallizes the vector database
        :param user_data_path: path to user's data folder
        :param device: device to load into
        """
        self.db_path = Path(str(user_data_path) + './rag.db')
        self.vector_path = Path(str(user_data_path) + './embeddings.pt')

        self.all_vectors = None
        self.device = device
        self.load_to_memory()
        self.init_sqlite()

    def init_sqlite(self):
        """
        initializes the user context database
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS chunks 
                            (id INTEGER PRIMARY KEY, 
                             doc_name TEXT, 
                             page_num INTEGER, 
                             content TEXT)''')

    def add_document_to_store(self, doc_name, chunks_list, new_embeddings_tensor):
        """
        adds document to storage
        :param doc_name: document name
        :param chunks_list: list of document chunks
        :param new_embeddings_tensor: embeddings tensor of document chunks
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(id) FROM chunks")
            result = cursor.fetchone()[0]
            start_id = (result + 1) if result is not None else 0
            for i, chunk_data in enumerate(chunks_list):
                chunk = chunk_data['chunk']
                cursor.execute(
                    "INSERT INTO chunks (id, doc_name, page_num, content) VALUES (?, ?, ?, ?)",
                    (start_id + i, doc_name, chunk_data['page_num'], chunk.contents)
                )
            conn.commit()

        self.append_vectors(new_embeddings_tensor)

    def append_vectors(self, new_vectors):
        """
        appends vectors to existing vector database
        :param new_vectors: cunks embeddings tensor to append
        """
        if self.vector_path.exists():
            old_vectors = torch.load(self.vector_path, map_location=self.device)
            combined_vectors = torch.cat([old_vectors, new_vectors], dim=0)
        else:
            combined_vectors = new_vectors

        torch.save(combined_vectors, self.vector_path)
        self.all_vectors = combined_vectors

    def load_to_memory(self):
        """
        load user storage to memory
        """
        try:
            self.all_vectors = torch.load(self.vector_path, map_location=self.device, weights_only=True)
            if len(self.all_vectors.shape) == 1:
                self.all_vectors = self.all_vectors.unsqueeze(0)
            return True

        except FileNotFoundError:
            print(f"Error: Could not find files at {self.vector_path}")
            return False

        except EOFError:
            self.all_vectors = None
            return False

    def search(self, query_embeddings, top_k=5, threshold=0.7):
        """
        preform similarity search through the vectordb using scaled dot product and fetch actual data from the database
        :param query_embeddings: prompt embedded
        :param top_k: amount of results to return
        :param threshold: minimum simillarity score
        :return: relevant context items
        """
        if self.all_vectors is None or self.all_vectors.shape[0] == 0:
            return []

        actual_k = min(top_k, self.all_vectors.shape[0])

        dot_scores = util.dot_score(query_embeddings, self.all_vectors)[0]
        top_results = torch.topk(dot_scores, k=actual_k)

        scores = top_results.values.tolist()
        indices = top_results.indices.tolist()

        valid_indices = []
        for i in range(len(scores)):
            if scores[i] >= threshold:
                valid_indices.append(indices[i])

        if not valid_indices:
            return []

        results = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            placeholders = ', '.join(['?'] * len(valid_indices))
            query = f"SELECT content, page_num FROM chunks WHERE id IN ({placeholders})"
            cursor.execute(query, valid_indices)
            results = cursor.fetchall()

        return results

    def list_documents(self):
        """
        lists all the documents the user owns
        :return: list of documnets
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT doc_name, COUNT(*) FROM chunks GROUP BY doc_name ORDER BY MIN(id)"
            )
            return [{"name": row[0], "chunks": row[1]} for row in cursor.fetchall()]

    def delete_document(self, doc_name):
        """
        deletes all chunks for a doc and rebuild the user's embedding tensor
        returns true if op complete
        :param doc_name: document name
        :return: bool
        """
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT id FROM chunks WHERE doc_name = ? ORDER BY id",
                (doc_name,)
            )
            ids_to_delete = set(r[0] for r in cur.fetchall())
            if not ids_to_delete:
                return False
            cur.execute("DELETE FROM chunks WHERE doc_name = ?", (doc_name,))
            conn.commit()

        if self.all_vectors is not None and self.all_vectors.shape[0] > 0:
            keep_mask = torch.ones(self.all_vectors.shape[0], dtype=torch.bool)
            for idx in ids_to_delete:
                if idx < keep_mask.shape[0]:
                    keep_mask[idx] = False
            self.all_vectors = self.all_vectors[keep_mask]
            torch.save(self.all_vectors, self.vector_path)

        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            rows = cur.execute(
                "SELECT id, doc_name, page_num, content FROM chunks ORDER BY id"
            ).fetchall()
            cur.execute("DELETE FROM chunks")
            cur.executemany(
                "INSERT INTO chunks (id, doc_name, page_num, content) VALUES (?, ?, ?, ?)",
                [(i, r[1], r[2], r[3]) for i, r in enumerate(rows)]
            )
            conn.commit()

        return True