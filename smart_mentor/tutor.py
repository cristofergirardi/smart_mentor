from .config.config_helper import ConfigHelper
from .orchestrator import SmartMentorOrchestrator
from .config import logging_config
from .observability.rouge_eval import RougeEval
from .file.smart_reader import SmartReader
from .file.smart_writer import SmartWriter
from typing import Final
import pandas as pd
import random

logger = logging_config.setup_logging()

class SmartMentor():

    ROUGE_METRICS: Final = ['rouge1', 'rouge2', 'rougeL']
    COLUMNS_METRICS: Final = ["h", "model", "metric", "precision", "recall", "f1_score"]

    def __init__(self, config: ConfigHelper):
        super().__init__()
        self.orchestrator = SmartMentorOrchestrator(config)      
        self.rouge = RougeEval()  
        logger.info("Smart Tutor is on!")
    
    def get_prompt(self, question, hypothesis):
        return self.orchestrator.prepare_prompt(question, hypothesis)
            
    def get_response_openai_by_prompt(self, prompt):
        return self.orchestrator.request_openai_by_prompt(prompt)

    def get_response_llama_by_prompt(self, prompt):
        return self.orchestrator.request_llama_by_prompt(prompt)

    def get_response_claudeai(self, question):
        return self.orchestrator.request_claudeai(question)
    
    def get_response_openai(self, question):
        return self.orchestrator.request_openai(question)

    def get_response_llama(self, question):
        return self.orchestrator.request_llama(question)
    
    def get_metrics(self, hypothesis: str, model:str, orig_data: str, predict: str) -> list:        
        self.rouge.get_scores(reference=orig_data, result=predict)
        metrics_list = []
        for metric in self.ROUGE_METRICS:
            precision, recall, fmeasure = self.rouge.get_score_by_name(metric_name=metric)
            metrics_list.append(
                {"h": hypothesis, "model": model, "metric": metric, "precision": precision, "recall": recall, "f1_score": fmeasure}
            )
        return metrics_list

    def add_new_row(self, df_orig : pd.DataFrame, new_rows: list):        
        for new_row in new_rows:
            df_orig.loc[len(df_orig)] = new_row
        return df_orig    
    
    @property
    def generate_random_numbers(self):
        random_numbers = [random.randint(0, 70) for _ in range(1)]
        return pd.DataFrame({'index': random_numbers})


if __name__ == "__main__":
    config = ConfigHelper()
    reader = SmartReader()
    writer = SmartWriter()
    tutor = SmartMentor(config)
    hypothesis = "h8"

    ## Creating file
    file_random = "smart_mentor/resources/random_numbers.csv"
    df_indexes = pd.DataFrame()
    if not reader.checkFile(file_random):
        writer.write(file_random, tutor.generate_random_numbers)
        df_indexes = reader.readFile(file_random)
    else:
        df_indexes = reader.readFile(file_random)

    df = reader.readFile("smart_mentor/resources/ground_truth_data.csv")
    df_selected = df.iloc[df_indexes['index']]
    df_metrics = pd.DataFrame(columns=tutor.COLUMNS_METRICS)

    for row in df_selected.itertuples(index=False): 
        # text_question = f"{row._2} \n {row._5} \n {row._6} \n {row._7} \n # Dica sobre a questão {row._9} \n Resposta: \n {row._20}"

        prompt = ""
        if len(str(row._9).replace("Dicas&Dicas","")) > 0:
            prompt = f"{row._2} \n {row._5} \n {row._6} \n {row._7} \n # Dica sobre a questão {row._9}"
        else:
            prompt = f"{row._2} \n {row._5} \n {row._6} \n {row._7}"
        reference = row._20

        new_prompt = tutor.get_prompt(hypothesis=hypothesis,
                                    question=prompt) 

        response = tutor.get_response_openai_by_prompt(prompt=new_prompt)
        print(f"#### OPENAI \n {response}") 
        list_metrics = tutor.get_metrics(hypothesis=hypothesis,
                                        model="openai",
                                        orig_data=reference,
                                        predict=response)
        for metrics in list_metrics:
            logger.info(f'From {metrics["metric"]} by rouge_score library -> Precision: {metrics["precision"]} Recall: {metrics["recall"]} fmeasure: {metrics["f1_score"]} ')

        response = tutor.get_response_llama_by_prompt(prompt=new_prompt)
        print(f"#### LLAMA \n {response}") 

        list_metrics = tutor.get_metrics(hypothesis=hypothesis,
                                        model="llama",
                                        orig_data=reference,
                                        predict=response)
        for metrics in list_metrics:
            logger.info(f'From {metrics["metric"]} by rouge_score library -> Precision: {metrics["precision"]} Recall: {metrics["recall"]} fmeasure: {metrics["f1_score"]} ')

    

