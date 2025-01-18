from urllib.request import urlopen
import pymupdf
from io import BytesIO
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from string import punctuation
from concurrent.futures import ThreadPoolExecutor


def process_pair(first, second, llm):
    if first == second: return None
    prompt = f'''
        Q: Is there an is-a relationship present between dog and mammal?
        A: Yes

        Q: Is there an is-a relationship present between vehicle and car?
        A: No

        Q: Is there an is-a relationship presnet between {first} and {second}?
        '''
    response = llm.invoke(prompt).content
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
  

def process_sample(metrics, sample) -> list[dict]:
    def process_metric(metric, sample):
        metric.measure(sample)
        result = {
            'name': metric.__name__,
            'score': metric.score,
            'input': sample.input,
            'output': sample.actual_output,
            'success': metric.is_successful(),
            'reason': metric.reason
        }
        return result

    sample_results = []
    futures = []
    with ThreadPoolExecutor(max_workers = len(metrics)) as pool:
        for metric in metrics:
            futures.append(pool.submit(process_metric, metric, sample))
        
        for f in futures:
            if f.result() is not None:
                sample_results.append(f.result())

    return sample_results


def rank_docs(queries_and_docs: list[tuple[str, str]], top_n: int) -> list:
    '''
    Ranks queries and documents using CrossEncoder

    Args:
        queries_and_docs (list[tuple[str, str]]): queries and documents
        top_n (int): number of docs to return

    Returns:
        list
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
    splitter = SentenceTransformersTokenTextSplitter(model_name = model)

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
    stemmer = PorterStemmer()
    punctuation = ['@', '%', '^', '*', '(', ')', '-', '_', '#', '~', '`', '\''] 

    text = ''.join([char for char in text if char not in punctuation])
    text = text.lower()

    words = word_tokenize(text = text)
    words = [stemmer.stem(word) for word in words]
    words = [word for word in words if word not in set(stopwords.words('english'))]

    return ' '.join(words)
