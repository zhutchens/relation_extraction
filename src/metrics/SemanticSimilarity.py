from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from sentence_transformers import SentenceTransformer
from src.utils import normalize_text

class SemanticSimilarity(BaseMetric):
    def __init__(self, threshold: float = 0.5, st_model: str = None) -> None:
        self.model = SentenceTransformer(st_model)
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase) -> float:
        expected = self.model.encode(normalize_text(test_case.expected_output))
        out = self.model.encode(normalize_text(test_case.actual_output))

        self.score = self.model.similarity(expected, out).item()
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
