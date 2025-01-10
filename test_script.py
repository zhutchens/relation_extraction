import sys
if len(sys.argv) <= 5 or len(sys.argv) >= 7:
    print('usage: <openai_model> <openai mode temperature> <sentence transformer model> <which_textbook> <what_to_test> <num_to_generate>')
    print('Options for which textbook: dsa_6114, cs_3190, dsa_2214')
    print('Options for what_to_test: concepts, outcomes, key_terms')
    print(f'num_to_generate: number of concepts, outcomes, or key_terms to generate for testing')
    sys.exit()

openai_model = sys.argv[1]
temp = sys.argv[2]
st_model = sys.argv[3]

textbook = sys.argv[4]
if textbook != 'dsa_6114' and textbook != 'cs_3190' and textbook != 'dsa_2214':
    print('invalid textbook arg. use dsa_2214, dsa_6114, or cs_3190')
    sys.exit()

testing = sys.argv[5]
if testing != 'concepts' and testing != 'outcomes' and testing != 'outcomes':
    print('invalid what_to_test arg. use concepts, outcomes, or key_terms')
    sys.exit()

num_generated = int(sys.argv[6])

from src.extractor import relationExtractor
from dotenv import load_dotenv
from os import getenv, environ
import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, ContextualPrecisionMetric, ContextualRecallMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

load_dotenv()
link = getenv(textbook) # Testing 2214 data structures textbook here
token = getenv('OPENAI_API_KEY')
environ['OPENAI_API_KEY'] = token
connection = getenv('connection_string')

if textbook == 'dsa_2214':
    chapters = [
    'Data Structures and Algorithms',
    'Mathematical Preliminaries',
    'Algorithm Analysis',
    'Lists, Stacks, and Queues',
    'Binary Trees',
    'Non-Binary Trees',
    'Internal Sorting',
    'File Processing and External Sorting',
    'Searching',
    'Indexing',
    'Graphs',
    'Lists and Arrays Revisited',
    'Advanced Tree Structures',
    'Analysis Techniques',
    'Lower Bounds',
    'Patterns of Algorithms',
    'Limits to Computation',
]
elif textbook == 'dsa_6114':
    chapters = ['The Role of Algorithms in Computing',
            'Getting Started',
            'Growth of Algoritmhs',
            'Divide-and-Conquer',
            'Probabilistic Analysis and Randomized Algorithms',
            'Heapsort',
            'Quicksort',
            'Sorting in Linear Time',
            'Medians and Order Statistics',
            'Elementary Data Structures',
            'Hash Tables',
            'Binary Search Trees',
            'Red-Black Trees',
            'Augmenting Data Structures',
            'Dynamic Programming',
            'Greedy Algorithms',
            'Amortized Analysis',
            'B-Trees',
            'Fibonacci Heaps',
            'van Emde Boas Trees',
            'Data Structure for Disjoint Sets',
            'Elementary Graph Algoritmhs',
            'Minimum Spanning Trees',
            'Single-Source Shortest Paths',
            'All-Pairs Shortest Paths',
            'Maximum Flow',
            'Multithreaded Algoritmhs',
            'Matrix Operations',
            'Linear Programming',
            'Polynomials and the FFT',
            'Number-Theoretic Algorithms',
            'String Matching',
            'Computational Geometry',
            'NP-Completeness',
            'Approximation Algorithms']
elif textbook == 'cs_3190':
    chapters = [
    'Meet Hadoop',
    'MapReduce',
    'The Hadoop Distributed Filesystem',
    'YARN',
    'Hadoop I/O',
    'Developing a MapReduce Application',
    'How MapReduce Works',
    'MapReduce Types and Formats',
    'MapReduce Features',
    'Setting up a Hadoop Cluster',
    'Administering Hadoop',
    'Avro',
    'Parquet',
    'Flume',
    'Sqoop',
    'Pig',
    'Hive',
    'Crunch',
    'Spark',
    'HBase',
    'Zookeeper',
    'Composable Data at Cerner',
    'Biological Data Science: Saving Lives with Software.',
    'Cascading'
]

CHUNK_SIZE = 3000
CHUNK_OVERLAP = 100

extractor = relationExtractor(link, 
                            token, 
                            chapters, 
                            connection, 
                            CHUNK_SIZE, 
                            CHUNK_OVERLAP, 
                            'DocumentEmbeddings', 
                            '2214_embeddings',
                            reset = True,
                            openai_model = sys.argv[0],
                            st_model = sys.argv[2],
                            temp = sys.argv[1])


if testing == 'concepts':
    concepts, retrieved = extractor.identify_concepts(num_generated)
elif testing == 'outcomes':
    outcomes, retrieved = extractor.identify_outcomes(num_generated)
else:
    terms, retrieved = extractor.identify_key_terms(num_generated)


def test_answer_relevancy():
    pass


def test_contextual_precision():
    pass


def test_contextual_recall():
    pass


def test_faithfulness():
    pass




