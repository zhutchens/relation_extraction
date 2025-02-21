import sys
import os 

if len(sys.argv) <= 6 or len(sys.argv) >= 9:
    print('usage: <llm> <sentence_transformer_model> <retriever_type> <course> <what_to_test> <num_to_generate> <threshold>')
    print('---Argument explanations---')
    print('\tllm: language model to use')
    print('\tsentence_transformer_model: sentence transformer model to use for embeddings')
    print('\tretriever type: type of retriever (vectordb or transformer)')
    print('\tcourse: course to use for evaluation, options: cs2, ccda')
    print('\twhat_to_test: what to test for evaluation from the following: concepts, outcomes')
    print('\tnum_to_generate: number of generated concepts or outcomes to get')
    print('\tthreshold: threshold for evaluation metrics to pass')

    sys.exit()

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

llm = sys.argv[1].lower()
st_model = sys.argv[2].lower()
retriever_type = sys.argv[3].lower()

if retriever_type != 'transformer' and retriever_type != 'vectordb':
    raise ValueError('retriever_type arg can only have the value transformer or vectordb')

course = sys.argv[4].lower()
if course != 'cs2' and course != 'ccda':
    print('invalid textbook arg. use cs2 or ccda')
    sys.exit()

testing = sys.argv[5].lower()
if testing != 'concepts' and testing != 'outcomes':
    print('invalid what_to_test arg. use concepts or outcomes')
    sys.exit()

num_generated = int(sys.argv[6])
threshold = float(sys.argv[7])

from dotenv import load_dotenv
from os import environ, getenv
load_dotenv()
environ['OPENCCDAAI_API_KEY'] = getenv('OPENAI_API_KEY')

from src.generator import RAGKGGenerator
from deepeval.metrics import AnswerRelevancyMetric, ContextualPrecisionMetric, ContextualRecallMetric, FaithfulnessMetric
from src.metrics import SemanticSimilarity, AnswerCorrectness
import pandas as pd
import os
# from src.llms import OpenAIModel, HuggingFaceLLM
from src.llms import OpenAIModel

link = getenv(course)
 
if course == 'cs2':
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

    data = pd.read_csv('data/dsa_clifford_a_shaffer_3_2_java.csv') # replace with path respective to cwd 
    # ex. if running from evaluate/ directory, use ../data/dsa_clifford_a_shaffer_3_2_java.csv

elif course == 'ccda':
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

    data = pd.read_csv('data/hadoop_the_definitive_guide.csv') # replace with path respective to cwd 
    # ex. if running from evaluate/ directory, use ../data/hadoop_the_definitive_guide.csv

else:
    raise ValueError('incorrect value! only use cs2 or ccda as the course arg')

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

if testing == 'concepts':
    actual = data['concepts']
else:
    actual = data['outcomes']

gen = RAGKGGenerator([chapters] if isinstance(chapters, str) else chapters, 
                            CHUNK_SIZE, 
                            CHUNK_OVERLAP, 
                            OpenAIModel(model_name = llm),
                            textbooks = link,
                            st_model = st_model,
                            retriever_type = retriever_type,
                            )


if testing == 'concepts':
    generated, retrieved = gen.identify_concepts(num_generated)
    actual = [c.split('::') for c in data['concepts']]
elif testing == 'outcomes':
    generated, retrieved = gen.identify_outcomes(num_generated)
    actual = [o.split('::') for o in data['outcomes']]
else:
    generated, retrieved = gen.identify_key_terms(num_generated)

metrics = [SemanticSimilarity(threshold, st_model = gen.embedding_model), AnswerRelevancyMetric(threshold, model = gen.llm), AnswerCorrectness(threshold, model = gen.llm), FaithfulnessMetric(model = gen.llm), ContextualPrecisionMetric(threshold, model = gen.llm), ContextualRecallMetric(threshold, model = gen.llm)]
results = gen.evaluate(testing, num_generated, actual, metrics = metrics)

if not os.path.exists('./results'):
    os.mkdir('./results/')
    os.chdir('./results/')
else:
    os.chdir('./results/')


with open(f'results_{course.upper()}_textbook_{testing}_{st_model}_{gen.llm.get_model_name()}.txt', 'w') as f:
    f.write('-' * 20 + '\n')
    f.write(f'OPENAI MODEL: {gen.llm.get_model_name()}\n')
    f.write(f'SENTENCE TRANSFORMER: {gen.embedding_model}\n')
    f.write(f'RETRIEVER: {gen.retriever.__name__}\n')
    f.write(f'TEXTBOOK: {course}\n')
    f.write(f'CHAPTERS TESTED: {gen.chapters}\n')
    f.write(f'CHUNK SIZE: {CHUNK_SIZE}\n')
    f.write(f'CHUNK OVERLAP: {CHUNK_OVERLAP}\n')
    f.write('-' * 20 + '\n')

    averages = {}
    for r in results:
        query = r['input']
        output = r['output']
        name = r['name']
        score = r['score']
        reason = r['reason']
        expected = r['expected']

        f.write('-' * 20 + '\n')
        f.write(f'METRIC: {name} ---> SCORE: {score} ---> {"FAILURE" if score < threshold else "SUCCESS"}\n')
        f.write('\n')
        f.write(f'REASON: {reason}\n')
        f.write('\n')
        f.write(f'QUERY: {query}\n')
        f.write('\n')
        f.write(f'EXPECTED {testing.upper()}: {expected}')
        f.write('\n')
        f.write(f'GENERATED {testing.upper()}: {output}\n')  
        f.write('-' * 20 + '\n')

        if name not in averages:
            averages[name] = score
        else:
            averages[name] += score

    for k in averages.keys():
        averages[k] /= len(gen.chapters) # calculate average metric scores across all chapters

    f.write(f'')
    f.write('AVERAGE SCORES:\n')
    for k, v in averages.items():
        f.write(f'{k}: {v}\n')

print(f'Evaluation complete.')
