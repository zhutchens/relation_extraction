# from pymongo import MongoClient
from langchain_core.documents import Document
# from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
# import chromadb
# from langchain_community.vectorstores.chroma import Chroma
from src.transformerEmbeddings import TransformerEmbeddings
from src.utils import chunk_doc
from langchain_core.vectorstores import InMemoryVectorStore
from src.utils import rank_docs
from src.utils import normalize_text
from rank_bm25 import BM25Okapi
# from pymongo import MongoClient


class RetrievalSystem:
    def __init__(self, 
                connection: str, 
                content: list[str] | str, 
                chunk_size: int,
                chunk_overlap: int, 
                db_name: str,
                collection_name: str,
                model: str,
                reset: bool = False):
        '''
        Construct a retrieval system using MongoDB Atlas

        Args:
            connection (str): connection string to MongoDB client
            content (list[str] | str): content to use in retrieval system. Available options are pdf paths, text string, or web links
            chunk_size (itn): number of characters to have in a chunk of the content
            chunk_overlap (int): number of characters to overlap between chunks
            db_name (str): name of MongoDB database
            collection_name (str): name of MongoDB collection
            model (str): sentence transformer model to use 
            reset (bool, default False): if True, drop all documents from collection before adding new content
        '''
        # self.db_name = db_name
        # self.client = MongoClient(connection)
        # self.collection = self.client[db_name][collection_name]

        self.docs = [Document(doc.page_content) for doc in chunk_doc(content, chunk_size, chunk_overlap, model)]
        self.docs_as_strings = [doc.page_content for doc in self.docs]
        
        # self.store = MongoDBAtlasVectorSearch(collection = self.collection, embedding = TransformerEmbeddings(model = model))
        self.store = InMemoryVectorStore.from_documents(self.docs, embedding = TransformerEmbeddings(model))
        self.bm25 = BM25Okapi(corpus = [doc.page_content for doc in self.docs])

        # self.store = Chroma(embedding_function = TransformerEmbeddings(model), persist_directory = './vectorstore')

        # self.embedder = TransformerEmbeddings(model)
        # self.store = self.embedder.embed_documents(self.docs_as_strings)

        # if reset:
            # self.remove_docs()
            # self.store.add_documents(documents = self.docs)
        #     self.store.persist()


    def pipeline(self, query: str, llm, num_queries: int = 4, top_n: int = 4) -> list[Document]:
        '''
        Retrieval pipeline from original query

        Args:
            query (str): original prompt
            llm: llm to use for generating queries
            num_queries (int, default 4): number of additional queries to generate
            top_n (int, default 4): top n documents to return

        Returns:
            list[Document]: top_n documents
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
                    
        semantic_queries = llm.invoke(semantic_q_prompt).content.split('::')
        keyword_queries = llm.invoke(keyword_q_prompt).content.split('::')

        retrieved_docs = []
        for q in semantic_queries:
            q_docs = self.semantic_search(q, top_n)

            for q_doc in q_docs:
                retrieved_docs.append((q, q_doc))

        for q in keyword_queries:
            k_docs = self.keyword_search(q, top_n)

            for k_doc in k_docs:
                retrieved_docs.append((q, Document(k_doc)))

        return rank_docs(retrieved_docs)[:top_n]


    def semantic_search(self, query: str, k: int = 4) -> list[Document]:
        '''
        Retrieves documents using semantic search

        Args:
            query (str): query to retriever
            k (int, default 4): number of relevant docs to retrieve

        Returns:
            list[Document]: top k similar documents
        '''
        # scores, indices = self.embedder.semantic_search(query, self.store, k)

        # docs = []
        # for score, index in zip(scores, indices):
        #     docs.append(self.docs[index])
        
        # return docs
        return self.store.similarity_search(query, k)


    def keyword_search(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        '''
        Retrieves documents using keyword search

        Args:
            query (str): query to retriever
            k (int, default 4): number of relevant docs to retrieve

        Returns:
            list[tuple[Document, float]]: documents and relevancy scores
        '''
        # keyword_docs = self.bm25.get_top_n(query, [doc.page_content for doc in self.docs], k)
        # keyword_scores = self.bm25.get_scores(query)

        # return zip(keyword_docs, keyword_scores)
        return self.bm25.get_top_n(query, [doc.page_content for doc in self.docs], k)


    def remove_docs(self):
        '''Remove all documents from collection'''
        self.collection.delete_many({})



