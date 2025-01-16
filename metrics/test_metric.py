from SemanticSimilarity import SemanticSimilarity
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from os import getenv, environ
from dotenv import load_dotenv


load_dotenv()
environ['OPENAI_API_KEY'] = getenv('OPENAI_API_KEY')

case = LLMTestCase(input = 'hello!!', actual_output = 'hey there!!')
# metric = AnswerRelevancyMetric(threshold = 0.9)
metric = SemanticSimilarity(st_model = 'all-MiniLM-L12-v2')

result = evaluate([case], [metric])
print(result)