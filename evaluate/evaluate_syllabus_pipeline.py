import sys
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

if len(sys.argv) > 5 or len(sys.argv) < 5:
    print('usage: <llm> <sentence transformer> <course to evaluate> <threshold>')
    print('available courses: CS2, CCDA (cloud computing for data analysis)')
    sys.exit()

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

llm = sys.argv[1]
st_model = sys.argv[2]
course = sys.argv[3].upper()
threshold = float(sys.argv[4])

from src.generator import RAGKGGenerator
from src.llms import OpenAIModel
from src.metrics import AnswerCorrectness, SemanticSimilarity
from deepeval.test_case import LLMTestCase

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# setup
if course == 'CS2': # for evaluating syllabus of CS2-DS course
    objectives = [
        'Compare and analyze basic building blocks of data structures, arrays, ArrayList, and linked lists',
        'Implement data structures (such as stacks, queues, generic lists, trees, hash maps, and graphs) based on an ADT, using arrays,ArrayList, or linked nodes, as appropriateChoose an appropriate', 
        'Choose an appropiate data structure for a given problem/situation (such as stacks, queues, generic lists, trees, hash maps, graphs)', 
        'Apply data structures (such as stacks, queues, generic lists, trees, hash maps, and graphs) to solve a given problem',
        'Use generics to ensure appropriate generalization of code',
        'Analyze the Big-O complexity of an algorithm or function',
        'Write and execute test cases for a class',
        'Trace and analyze recursive algorithm',
    ]

    topics = [
        'Collections, Abstract Data Types, Interfaces, Generics',
        'Unit Testing, Exception Handling, JavaDoc',
        'Big O Notation & Analysis',
        'Arrays & ArrayLists',
        'Stacks & Queues',
        'Comparing, Sorting & Searching',
        'Linked Nodes and Linked Structures',
        'Lists',
        'Recursion & Recursive Searching & SortingTrees & Heaps',
        'Graphs',
        'Hash Functions, Hash Tables, and Hash Maps',
    ]

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


    gen = RAGKGGenerator(
        chapters,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        OpenAIModel(model_name = llm),
        os.getenv('cs2'),
        '../data/Syllabusfor202380-Fall2023-ITSC-2214-001-DataStructuresandAlgorithms.pdf', # change based on current working directory
        st_model = st_model
    )

    

elif course == 'CCDA':
    objectives = [
        'To enable students to demonstrate an understanding of the basic concepts of cloud platforms and tools for data analysis',
        'To enable students to apply these concepts to real problems through extensive hands-on experience in deploying and analyzing data using cloud tools discovering actionable insights from large-scale data or develop predictive applications',
        'To enable students to demonstrate programming skills for cloud platforms',
    ]

    topics = [
        'Distributed Computing and Clouds',
        'Data Analysis Algorithms (clustering, classification)',
        'Hadoop',
        'HDFS',
        'YARN',
        'MapReduce',
        'Pig',
        'Hive',
        'Spark',
        'Information Retrieval',
        'Web Search',
        'Page Rank',
    ]

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

    gen = RAGKGGenerator(chapters, 
        CHUNK_SIZE, 
        CHUNK_OVERLAP, 
        OpenAIModel(model_name = llm), 
        os.getenv('ccda'), 
        '../data/Syllabus for 202380-Fall 2023-ITCS-3190-001-Cloud Comp for Data Analysis.pdf', # replace this with respect to current working directory
        st_model = st_model
    )

else:
    raise ValueError('incorrect value! only use CS2 or CCDA as the course arg')


gen_topics = gen.identify_main_topics()
gen_objectives = gen.objectives_from_syllabus()

actual = [gen_topics, gen_objectives]
expected = [topics, objectives]
metrics = [AnswerCorrectness(threshold, gen.llm), SemanticSimilarity(threshold, gen.embedding_model)]

inp = 'No input provided'

if os.getcwd().endswith('rag'):
    save_to = f'./results/results_{course}_syllabus_{st_model}_{llm}.txt'
else:
    save_to = f'../results/results_{course}_syllabus_{st_model}_{llm}.txt'

with open(save_to, 'w') as f:
    f.write('-' * 30 + '\n')
    f.write(f'OPENAI MODEL: {gen.llm.get_model_name()}\n')
    f.write(f'SENTENCE TRANSFORMER: {gen.embedding_model}\n')
    f.write(f'COURSE: {course}\n')
    f.write(f'CHUNK SIZE: {CHUNK_SIZE}\n')
    f.write(f'CHUNK OVERLAP: {CHUNK_OVERLAP}\n')
    f.write('-' * 30 + '\n')

    for i in range(len(actual)):
        for metric in metrics:
            test_case = LLMTestCase(inp, ' '.join(actual[i]), ' '.join(expected[i]))

            metric.measure(test_case)

            f.write(f'Metric: {metric.__name__}\n')
            f.write(f'Testing: {"Topics" if i == 0 else "Objectives"}\n')
            f.write(f'Score: {metric.score}\n')
            f.write(f'Reason: {metric.reason}\n')
            f.write(f'Actual: {actual[i]}\n')
            f.write(f'Expected: {expected[i]}\n')
            f.write('-' * 30 + '\n')

print('Done')





