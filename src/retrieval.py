# from pymongo import MongoClient
from langchain_core.documents import Document
# from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
# import chromadb
from langchain_community.vectorstores.chroma import Chroma
from src.transformerEmbeddings import TransformerEmbeddings
from src.utils import chunk_doc
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.retrievers import BM25Retriever
from src.utils import rank_docs
# from src.utils import normalize_text



class RetrievalSystem:
    def __init__(self, 
                connection: str, 
                content: str, 
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
            content (str): content to use in retrieval system. Available options are pdf paths, text string, or web links
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

        # chunk and put text in documents 
        # idk the best way to do this
        try:
            chunks = chunk_doc(chunk_size, chunk_overlap, link = content, model = model)
        except Exception:
            try:
                chunks = chunk_doc(chunk_size, chunk_overlap, document = content, model = model)
            except Exception:
                chunks = chunk_doc(chunk_size, chunk_overlap, pdf_path = content, model = model)

        # # normalize text chunks
        # # chunks = [normalize_text(chunk) for chunk in chunks]

        # # create documents
        self.docs = [Document(page_content = chunk) for chunk in chunks]

        # if reset:
        #     self.remove_docs()
        
        # self.store = MongoDBAtlasVectorSearch(collection = self.collection, embedding = TransformerEmbeddings(model = model))
        self.store = InMemoryVectorStore.from_documents(self.docs, embedding = TransformerEmbeddings(model))
        self.bm25 = BM25Retriever.from_documents(self.docs)
        # self.store = Chroma(collection = 'embeddings', embedding_function = TransformerEmbeddings(model), persist_directory = './embeddings')

        # if reset:
        # self.store.add_documents(documents = self.docs)
            # self.store.persist()

    def pipeline(self, query: str, context: str, llm, num_queries: int = 4, top_n: int = 4):
        '''
        Retrieval pipeline from original query

        Args:
            query (str): original prompt
            context (str): (something)
            llm: llm to use for generating queries
            num_queries (int, default 4): number of additional queries to generate
            top_n (int, default 4): top n documents to return

        Returns:
            list[Document]: top_n documents
        '''
        q_prompt = f'''
                    You are tasked with enhancing this {query} for a retrieval-augmented pipeline. Output {num_queries} additional queries.
                    You can find relevant context here: {context}.

                    Output Format:
                    query_1::query_2::query_3::query_4...::query_{num_queries}
                    '''
        
        queries = llm.invoke(q_prompt).content.split('::')

        retrieved_docs = []
        for q in queries:
            semantic_docs, keyword_docs = self.invoke(q)
            
            for s in semantic_docs:
                if s not in retrieved_docs:
                    retrieved_docs.append((q, s))

            for k in keyword_docs:
                if k not in retrieved_docs:
                    retrieved_docs.append((q, k))

        # perform reranking and return
        ranked_docs = rank_docs(retrieved_docs, top_n)

        # convert scores/docs to only docs and return
        return [t[0][1] for t in ranked_docs]


    def invoke(self, query: str, k: int = 4) -> tuple[list[Document], list[Document]]:
        '''
        Retrieves documents using hybrid search with keywords and semantic similarity

        Args:
            query (str): query to retriever
            k (int, default 4): number of relevant docs to retrieve

        Returns:
            tuple[list[Document], list[Document]]: documents from semantic search, documents from keyword search
        '''
        # query = normalize_text(text = query)
        # print(f'Query after normalization:', query)
        keyword_results = self.bm25.invoke(query)
        similarity_results = self.store.similarity_search(query, k)

        return (similarity_results, keyword_results)



    def remove_docs(self):
        '''Remove all documents from collection'''
        self.collection.delete_many({})



