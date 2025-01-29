from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
import torch


class TransformerEmbeddings(Embeddings):
    def __init__(self, model: str):
        '''
        Constructor for transformer embeddings

        Args:
            model (str): sentence transformer to use
        '''
        self.model = SentenceTransformer(model)


    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        '''
        Embed documents 

        Args:
            texts (list[str]): documents to embed

        Returns:
            np.ndarray: embeddings
        '''
        return [self.model.encode(text).flatten().tolist() for text in texts]


    def embed_query(self, text: str) -> list[float]:
        '''
        Embed query

        Args:
            text (str): query to embed

        Returns:
            np.ndarray: embedded query
        '''
        return self.model.encode(text).flatten().tolist()


    def semantic_search(self, query: str, corpus_embeddings: list[list[float]], k: int):
        '''
        Semantic search using transformer

        Args:
            query (str): query to find similar documents for
            corpus_embeddings (list[list[float]]): list of embedded documents
            k (int): top k documents to return

        Returns:
            tuple[list, list]: list of scores and associated indices in corpus embeddings
        '''
        encoded_query = self.embed_query(query)
        similarity_scores = self.model.similarity(encoded_query, corpus_embeddings)[0]

        return torch.topk(similarity_scores, k)
