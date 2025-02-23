from src.utils import normalize_text, chunk_doc, rank_docs
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from deepeval.models import DeepEvalBaseLLM


class TransformerRetriever:
    def __init__(self, content: str | list[str], chunk_size: int, chunk_overlap: int, model: str) -> None:
        '''
        Construct a retrieval system using sentence transformers 

        Args:
            content (list[str] | str): content to use in retrieval system. Available options are pdf paths, text string, or web links
            chunk_size (itn): number of characters to have in a chunk of the content
            chunk_overlap (int): number of characters to overlap between chunks
            model (str): sentence transformer model to use 
        '''
        self.embedder = SentenceTransformer(model)

        self.chunks_as_docs = chunk_doc(content, chunk_size, chunk_overlap)
        # normalizing text here for for best keyword results from bm25
        self.chunks_as_strings = [doc.page_content.replace('\n\n', '') for doc in self.chunks_as_docs]

        self.bm25 = BM25Okapi([self.embedder.tokenizer.tokenize(string) for string in self.chunks_as_strings])

        self.corpora_embeddings = self.embedder.encode(self.chunks_as_strings)

    
    def pipeline(self, chapter_name: str, llm: DeepEvalBaseLLM) -> list[str]:
        '''
        Retrieval pipeline using hybrid search 

        Args:
            chapter_name (str): name of chapter to retrieve docs for 
            llm: llm to use for generating queries

        Returns:
            list[str]: list of top n documents

        ''' 
        chapter_prompt = f'''
                        Provide 10 questions for the topics of {chapter_name}. These should be very generic and simple questions. Provide your response as a list of questions using this format: question_1,question_2,question_3,question_4,question_5
                        
                        Additionally, the questions should only be on topics of {chapter_name}, not referencing anything residing outside that scope.

                        For instance, if the chaper name is Sorting, some example questions may be:
                        1. What is insertion sort?
                        2. What is bubble sort?
                        3. What is quick sort?
                        4. What is merge sort?
                        5. What is selection sort?

                        As you can see, these example questions ask specfically about topics that object oriented programming cover, but do not reference anything outside that scope. 
                        '''

        response = llm.generate(chapter_prompt)
        
        retrieved = set()

        for r in response.split(','):
            s_docs = self.semantic_search(normalize_text(r)) # returns list of dictionaries 
            k_scores = self.keyword_search(normalize_text(r)) # returns ndarray of floats
    
            s_docs = [self.chunks_as_strings[d['corpus_id']] for d in s_docs[0]] # get docs by corpus id (index)
    
            k_docs = sorted(zip(self.chunks_as_strings, k_scores), key = lambda item: item[1], reverse = True) # sort documents by top keyword matches
    
            # k_docs from sorted(zip) is list of tuples so get only strings now
            k_docs = [k_doc[0] for k_doc in k_docs]
    
            for s_doc, k_doc in zip(s_docs, k_docs):
                retrieved.add((r, s_doc))
                retrieved.add((r, k_doc))

        return rank_docs(list(retrieved))


    def semantic_search(self, query: str, k: int = 10):
        '''
        Retrieves documents using semantic search

        Args:
            query (str): query to retriever
            k (int, default 10): number of relevant docs to retrieve

        Returns:
            list[dict[str, int | float]]
        '''
        embedded_query = self.embedder.encode(query, convert_to_tensor = True)
        return util.semantic_search(embedded_query, self.corpora_embeddings, top_k = k)


    def keyword_search(self, query: str):
        '''
        Retrieves documents using keyword search

        Args:
            query (str): query to retriever
            k (int, default 4): number of relevant docs to retrieve

        Returns:
            list[float]
        '''
        return self.bm25.get_scores(self.embedder.tokenizer.tokenize(normalize_text(query))).tolist()


    @property
    def __name__(self):
        return 'TransformerRetriever'