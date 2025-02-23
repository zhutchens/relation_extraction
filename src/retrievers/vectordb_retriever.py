from langchain_core.documents import Document
from src.transformerEmbeddings import TransformerEmbeddings
from src.utils import chunk_doc
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from src.utils import rank_docs
from rank_bm25 import BM25Okapi
from deepeval.models import DeepEvalBaseLLM


class VectorDBRetriever:
    def __init__(self, 
                content: list[str] | str, 
                chunk_size: int,
                chunk_overlap: int, 
                model: str,
                ):
        '''
        Construct a retrieval system using MongoDB Atlas

        Args:
            content (list[str] | str): content to use in retrieval system. Available options are pdf paths, text string, or web links
            chunk_size (itn): number of characters to have in a chunk of the content
            chunk_overlap (int): number of characters to overlap between chunks
            model (str): sentence transformer model to use 
        '''
        self.docs = chunk_doc(content, chunk_size, chunk_overlap, 'transformer')
        self.docs_as_strings = [doc.page_content for doc in self.docs]
        
        client = QdrantClient(':memory:') 
        # change size param to match sentence transformer output size (ex, 768 for msmarco-distilbert-base-tas-b)
        client.create_collection('my_collection', vectors_config = VectorParams(size = 768, distance = Distance.COSINE))
        self.store = QdrantVectorStore(client, 'my_collection', embedding = TransformerEmbeddings(model))
        self.store.add_documents(self.docs)

        self.bm25 = BM25Okapi(corpus = [doc.page_content for doc in self.docs])


    def pipeline(self, query: str, llm: DeepEvalBaseLLM, num_queries: int = 4) -> list[str]:
        '''
        Retrieval pipeline from original query

        Args:
            query (str): original prompt
            llm : llm to use for generating queries
            num_queries (int, default 4): number of additional queries to generate

        Returns:
            list[str]: top_n documents
        '''
        semantic_q_prompt = f'''
                    You are tasked with enhancing this query: {query} for a semantic search retrieval-augmented pipeline. Output {num_queries} additional queries.

                    Output Format:
                    query_1::query_2::query_3::query_4...::query_{num_queries}
                    '''

        keyword_q_prompt = f'''
                    You are tasked with enhancing this query: {query} for a keyword search retrieval-augmented pipeline. Output {num_queries} keywords.

                    Output Format:
                    query_1::query_2::query_3::query_4...::query_{num_queries}
                    '''
                    
        semantic_queries = llm.generate(semantic_q_prompt).split('::')
        keyword_queries = llm.generate(keyword_q_prompt).split('::')

        retrieved_docs = set()
        for q in semantic_queries:
            q_docs = self.semantic_search(q)

            for q_doc in q_docs:
                retrieved_docs.add((q, q_doc.page_content))

        for q in keyword_queries:
            k_docs = self.keyword_search(q)

            for k_doc in k_docs:
                retrieved_docs.add((q, k_doc))

        return rank_docs(list(retrieved_docs))


    def semantic_search(self, query: str, k: int = 10) -> list[Document]:
        '''
        Retrieves documents using semantic search

        Args:
            query (str): query to retriever
            k (int, default 10): number of relevant docs to retrieve

        Returns:
            list[Document]: top k similar documents
        '''
        return self.store.similarity_search(query, k)


    def keyword_search(self, query: str, k: int = 10) -> list[tuple[Document, float]]:
        '''
        Retrieves documents using keyword search

        Args:
            query (str): query to retriever
            k (int, default 10): number of relevant docs to retrieve

        Returns:
            list[tuple[Document, float]]: documents and relevancy scores
        '''
        return self.bm25.get_top_n(query, [doc.page_content for doc in self.docs], k)


    @property
    def __name__(self):
        return 'VectorDBRetriever'


