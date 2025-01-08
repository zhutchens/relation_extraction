from urllib.request import urlopen
import pymupdf
from io import BytesIO
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


# Code taken from 
def rank_docs(queries_and_docs: list[tuple[str, str]], top_n: int) -> list[Document]:
    '''
    Ranks queries and documents using CrossEncoder

    Args:
        queries_and_docs (list[tuple[str, str]]): queries and documents
        top_n (int): number of docs to return

    Returns:
        list: list of top_n documents
    '''
    model = CrossEncoder(model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2')
    scores = model.predict([(t[0], t[1].page_content) for t in queries_and_docs])

    return sorted(list(zip(queries_and_docs, scores)), key = lambda x: x[1], reverse = True)[:top_n]


def chunk_doc(chunk_size: int, chunk_overlap: int, model: str, link: str = None, document: str = None, pdf_path: str = None) -> list[str]:
    '''
    Chunks an entire document

    Args:
        chunk_size (int): number of characters in a chunk
        chunk_overlap (int): number characters that overlap between chunks
        model (str): sentence transformer model to use 
        link (str, default None): a web link to chunk
        document (str, default None): entire document string to chunk
        pdf_path (str, default None): a pdf path to chunk

    Returns:
        list[str]: list of text chunks
    '''
    splitter = SentenceTransformersTokenTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap, model_name = model)

    if link is not None:
        content = urlopen(link).read()
        doc = pymupdf.open('pdf', BytesIO(content))
        text = ' '.join([page.get_text() for page in doc])
        text = text.replace('\n', ' ')
        return splitter.split_text(text)

    elif document is not None:
        text = document.replace('\n', ' ')
        return splitter.split_text(text)

    elif pdf_path is not None:
        doc = pymupdf(pdf_path)
        text = ' '.join([page.get_text() for page in doc])
        text = text.replace('\n', ' ')
        return splitter.split_text(text)

    else:
        raise ValueError(f'One of the following args must have a value: link, pdf_path, or document.')
        

def normalize_text(text: str) -> str:
    '''
    Normalize and clean text 

    Args:
        text (str): text to clean

    Returns:
        str: cleaned text
    '''
    lemmatizer = WordNetLemmatizer()
    punctuation = ['@', '%', '^', '*', '(', ')', '-', '_', '#', '~', '`', '\''] 

    text = ''.join([char for char in text if char not in punctuation])
    text = text.lower()

    words = word_tokenize(text = text)
    words = [lemmatizer.lemmatize(word) for word in words]
    words = [word for word in words if word not in set(stopwords.words('english'))]

    return ' '.join(words)
