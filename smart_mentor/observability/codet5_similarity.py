from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.nn.functional import cosine_similarity
from ..config import logging_config
import torch

logger = logging_config.setup_logging()

class CodeT5Similarity():

    # Using torch to calculate the similarity between generate code and ground-truth
    # torch is used to work with neural networks

    def __init__(self):
        model_name = "Salesforce/codet5-base" 
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def _generateTokenizer(self, ground_truth:str, result:str):
        # Tokenize both snippets
        gen_tokens = self.tokenizer(result, return_tensors="pt", truncation=True, padding=True)
        truth_tokens = self.tokenizer(ground_truth, return_tensors="pt", truncation=True, padding=True)
        return truth_tokens, gen_tokens
    
    def _get_embedding(self, ground_truth:str, result:str):
        truth_tokens, gen_tokens = self._generateTokenizer(ground_truth, result)
        with torch.no_grad():
            gen_embedding = self.model.encoder(**gen_tokens).last_hidden_state.mean(dim=1)
            truth_embedding = self.model.encoder(**truth_tokens).last_hidden_state.mean(dim=1)
        return truth_embedding, gen_embedding
    
    def get_similarity(self, ground_truth:str, result:str):
        truth_embedding, gen_embedding = self._get_embedding(ground_truth, result)
        similarity = cosine_similarity(gen_embedding, truth_embedding).item()
        return similarity

