from ..config import logging_config
from ..config.config_helper import ConfigHelper
from ..observability.rouge_eval import RougeEval
from ..orchestrator import SmartMentorOrchestrator
from ..file.smart_reader import SmartReader
from ..file.smart_writer import SmartWriter
import time
import pandas as pd
import random
from typing import Final

logger = logging_config.setup_logging()

class EvaluationModels():

    ROUGE_METRICS: Final = ['rouge1', 'rouge2', 'rougeL']
    COLUMNS_METRICS: Final = ["h", "model", "metric", "precision", "recall", "f1_score"]

    def __init__(self, config: ConfigHelper):
        super().__init__()
        self.orchestrator = SmartMentorOrchestrator(config)    
        self.rouge = RougeEval()    
        logger.info("Evaluation Models is on!")
    
    def get_prompt(self, question):
        return self.orchestrator.prepare_prompt(question)

    def get_response_openai(self, prompt):
        return self.orchestrator.request_openai_by_prompt(prompt)
    
    def get_response_claudeai(self, question):
        return self.orchestrator.request_claudeai(question)

    def get_response_llama(self, prompt):
        return self.orchestrator.request_llama_by_prompt(prompt)

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
        random_numbers = [random.randint(0, 13109) for _ in range(10)]
        return pd.DataFrame({'index': random_numbers})


if __name__ == "__main__":
    config = ConfigHelper()
    reader = SmartReader()
    writer = SmartWriter()
    models = EvaluationModels(config)

    ## Creating file
    file_random = "src/resources/random_numbers.csv"
    df_indexes = pd.DataFrame()
    if not reader.checkFile(file_random):
        writer.write(file_random, models.generate_random_numbers)
        df_indexes = reader.readFile(file_random)
    else:
        df_indexes = reader.readFile(file_random)

    df = reader.readFile("src/resources/Python Programming Questions Dataset.csv")
    df_selected = df.iloc[df_indexes['index']]
    df_metrics = pd.DataFrame(columns=models.COLUMNS_METRICS)

    hypothese = "h0"
    file_written = f"src/resources/Metrics_{hypothese}.csv"
    ## Creating file
    try:
        reader.removeFile("src/resources/Metrics.csv")
    except FileNotFoundError as e:
        logger.error("File does not found")
    
    writer.write(file_written, df_metrics)

    # how many times I'm going to call the models
    for count in range(2):
        logger.info(f"###### Counting {count} times ######")
        df_metrics = pd.DataFrame(columns=models.COLUMNS_METRICS)
        for row in df_selected.itertuples(): 

            user_question = ""

            if row.Input is not None:
                user_question = row.Instruction
            else:
                user_question = f'{row.Instruction} \n Using this data like an input: {row.Input}'

            prompt = models.get_prompt(user_question)

            response_openai = models.get_response_openai(prompt) 

            list_metrics = models.get_metrics(hypothese, "openai", row.Output, response_openai)
            df_metrics = models.add_new_row(df_metrics, list_metrics)

            response_llama = models.get_response_llama(prompt)

            list_metrics = models.get_metrics(hypothese, "llama", row.Output, response_llama)
            df_metrics = models.add_new_row(df_metrics, list_metrics)
            
            logger.info(f'Finished the index {row.Index}')
            time.sleep(60)

        ## Appending dataframe
        writer.write(file_written, df_metrics)            