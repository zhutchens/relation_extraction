from pymongo import MongoClient
from langchain_core.documents import Document
# from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
# import chromadb
from langchain_community.vectorstores.chroma import Chroma
from src.transformerEmbeddings import TransformerEmbeddings
from src.utils import chunk_doc
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore


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
        self.store = InMemoryVectorStore(embedding = TransformerEmbeddings(model = model))
        # self.store = Chroma(collection = 'embeddings', embedding_function = TransformerEmbeddings(model), persist_directory = './embeddings')

        # if reset:
        self.store.add_documents(documents = self.docs)
            # self.store.persist()


    def invoke(self, query: str, k: int = 4) -> list[Document]:
        '''
        Retrieve relevant docs using query

        Args:
            query (str): query to retriever
            k (int, default 4): number of relevant docs to retrieve

        Returns:
            list[Document]: list of relevant documents
        '''
        # query = normalize_text(text = query)
        # print(f'Query after normalization:', query)
        return self.store.similarity_search(query = query, k = k)


    def remove_docs(self):
        '''Remove all documents from collection'''
        self.collection.delete_many({})



