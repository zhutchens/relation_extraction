from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from sentence_transformers import SentenceTransformer


class SemanticSimilarity(BaseMetric):

    def __init__(self, threshold: float = 0.5, st_model: str = None) -> None:
        print(f'Inside semantic similarity, st_model is {st_model}')
        self.model = SentenceTransformer(st_model)
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase) -> float:
        input_embedding = self.model.encode(test_case.input)
        output_embedding = self.model.encode(test_case.actual_output)

        self.score =  self.model.similarity(input_embedding, output_embedding).item()
        self.success = self.score >= self.threshold
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
        return 'SemanticSimilarity'