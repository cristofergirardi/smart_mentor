from rouge_score import rouge_scorer
import evaluate
import pandas as pd

class RougeEval():

    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.scorer_hf = evaluate.load("rouge")
        self.scores = {}
        self.scores_hf = pd.DataFrame()

    def get_scores(self, reference: str, result: str):
        self.scores = self.scorer.score(reference, result)
        self.scores_hf = self.scorer_hf.compute(predictions=[result], references=[reference], use_aggregator=False)
    
    def get_score_by_name(self, metric_name: str):
        for metric, score in self.scores.items():
            if metric == metric_name:
                return score.precision, score.recall, score.fmeasure

    def get_score_hf_by_name(self, metric_name: str):
        return self.scores_hf[metric_name]
    
