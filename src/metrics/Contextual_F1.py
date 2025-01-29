from deepeval.metrics import BaseMetric, ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase

class Contextual_F1(BaseMetric):
    # f1 = 2 * (precision * recall / precision + recall)

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase) -> float:
        precision_score = ContextualPrecisionMetric().measure(test_case).score
        recall_score = ContextualRecallMetric().measure(test_case).score

        self.score = 2 * ((precision_score * recall_score) / (precision_score + recall_score))
        self.successful = self.score >= self.threshold
        return self.score

    def a_measure(self, test_case: LLMTestCase, _show_indicator: bool) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            return self.success

    @property
    def __name__(self):
        return 'Contextual_F1'

        
        