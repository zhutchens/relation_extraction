from deepeval.metrics import GEval, BaseMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models import DeepEvalBaseLLM
from src.llms import OpenAIModel

class AnswerCorrectness(BaseMetric):
    def __init__(self, threshold: float = 0.5, model: DeepEvalBaseLLM = OpenAIModel()) -> None:
        self.threshold = threshold
        self.model = model

    def measure(self, test_case: LLMTestCase) -> float:
        metric = GEval(
            name = "AnswerCorrectness",
            evaluation_steps = [    
                "Determine whether the actual output is factually correct based on the expected output.",
                "Differences in grammar is OK.",
            ],
            evaluation_params = [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            model = self.model,
            threshold = self.threshold,
        )   
        
        metric.measure(test_case)
        self.success = metric.is_successful()
        self.reason = metric.reason
        self.score = metric.score
        return self.score

    async def a_measure(self, test_case: LLMTestCase, _show_indicator: bool) -> float: 
        return self.measure(test_case)

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            return self.success

    @property
    def __name__(self):
        return "Answer Correctness"

