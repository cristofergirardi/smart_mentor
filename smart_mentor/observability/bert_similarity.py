from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import cosine_similarity
from ..config import logging_config

logger = logging_config.setup_logging()

class BertSimilarity():

    # Using torch to calculate the similarity between generate code and ground-truth
    # torch is used to work with neural networks

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = AutoModel.from_pretrained("microsoft/codebert-base")

    def _generateTokenizer(self, ground_truth:str, result:str):
        # Tokenize both snippets
        gen_tokens = self.tokenizer(result, return_tensors="pt", truncation=True, padding=True)
        truth_tokens = self.tokenizer(ground_truth, return_tensors="pt", truncation=True, padding=True)
        return gen_tokens, truth_tokens
    
    def _get_embedding(self, ground_truth:str, result:str):
        gen_tokens, truth_tokens = self._generateTokenizer(ground_truth, result)
        gen_embedding = self.model(**gen_tokens).last_hidden_state.mean(dim=1)
        truth_embedding = self.model(**truth_tokens).last_hidden_state.mean(dim=1)
        return gen_embedding, truth_embedding
    
    def get_similarity(self, ground_truth:str, result:str):
        gen_embedding, truth_embedding = self._get_embedding(ground_truth, result)
        similarity = cosine_similarity(gen_embedding, truth_embedding).item()
        return similarity

