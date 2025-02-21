from langchain_text_splitters import CharacterTextSplitter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langchain_core.documents import Document
from string import punctuation
from sentence_transformers import CrossEncoder
from unstructured.partition.auto import partition
from unstructured.cleaners.core import clean_extra_whitespace, clean_non_ascii_chars
import os
import validators
from string import punctuation


def process_pair(first, second, llm):
    if first == second: return None
    prompt = f'''
        Q: Is there an is-a relationship present between dog and mammal?
        A: Yes

        Q: Is there an is-a relationship present between vehicle and car?
        A: No

        Q: Is there an is-a relationship present between {first} and {second}?
        '''
    response = llm.generate(prompt)
    return (first, second) if 'yes' in response.lower() else None


def clean(text):
    text = ''.join([char for char in text if char not in punctuation])
    text = text.strip()
    return text


def create_concept_graph_structure(param_list: list) -> dict:
    return_dict = {}

    for list_itm in param_list:
        return_dict[list_itm] = []

    return return_dict


def rank_docs(queries_and_docs) -> list[Document]:
    '''
    Ranks queries and documents using reciprocal rank fusion

    Args:
        scores_and_docs (list[tuple[Document, float]]): documents and relevancy scores
        top_n (int): number of docs to return

    Returns:
        list[Document]: reranked documents 
    '''
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    scores = model.predict(queries_and_docs)
    
    sorted_docs = sorted(list(zip(queries_and_docs, scores)), key = lambda x: x[1], reverse = True)
    return [doc for (query, doc), score in sorted_docs]


def chunk_doc(documents: list[str] | str, chunk_size: int, chunk_overlap: int) -> list[Document]:
    '''
    Chunks an entire document

    Args:
        documents (list[str] | str): documents to chunk
        chunk_size (int): number of characters in a chunk
        chunk_overlap (int): number characters that overlap between chunks

    Returns:
        list[str]: list of text chunks
    '''
    # def process_loader(loader):
    #     content = ''
    #     for chunk in loader.load():
    #         content += chunk.page_content + ' '

    #     return content
    
    splitter = CharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)

    # all_doc_content = ''

    # if isinstance(documents, list): # if documents is a list (could contain a mix of files, links, and python strings)
    #     document_dict = {'urls': [], 'files': [], 'strings': []}
    #     for doc in documents:
    #         if os.path.exists(doc):
    #             document_dict['files'].append(doc)
    #         elif validators.url(doc):
    #             document_dict['urls'].append(doc)
    #         else:
    #             document_dict['strings'].append(doc)

        # url_loader = UnstructuredURLLoader(document_dict['urls'])
        # file_loader = UnstructuredFileLoader(document_dict['files'])

        # url_loader = []
        # for url in document_dict['urls']:
        #     url_elements = partition(url = url)
        #     url_loader.extend([Document(page_content = element) for element in url_elements])

        # file_loader = []
        # for f in document_dict['files']:
        #     file_elements = partition(filename = f)
        #     file_loader.extend([Document(page_content = element) for element in file_elements])

        # string_loader = [Document(page_content = value) for value in document_dict['strings']]

        # for loader in [url_loader, file_loader, string_loader]:
        #     all_doc_content += process_loader(loader)

        # return splitter.create_documents(all_doc_content)

    if os.path.exists(documents): # if documents is a single file
        elements = partition(filename = documents)
        return splitter.create_documents([str(el) for el in elements])

    elif validators.url(documents): # if documents is a single url
        elements = partition(url = documents)
        return splitter.create_documents([str(el) for el in elements])

    else: # if documents is an actual python string
        return splitter.create_documents(documents)


def normalize_text(text: str) -> str:
    '''
    Normalize and clean text 

    Args:
        text (str): text to clean

    Returns:
        str: cleaned text
    '''
    text = clean_extra_whitespace(text)
    text = clean_non_ascii_chars(text)

    l = WordNetLemmatizer()

    text = ''.join([char for char in text if char not in punctuation])
    text = text.lower()

    words = word_tokenize(text = text)
    words = [l.lemmatize(word) for word in words]
    words = [word for word in words if word not in set(stopwords.words('english'))]

    return ' '.join(words)
