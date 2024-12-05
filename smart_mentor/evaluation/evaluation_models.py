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
import json

logger = logging_config.setup_logging()

class EvaluationModels():

    ROUGE_METRICS: Final = ['rouge1', 'rouge2', 'rougeL']
    COLUMNS_METRICS: Final = ["h", "model", "metric", "precision", "recall", "f1_score"]

    def __init__(self, config: ConfigHelper):
        super().__init__()
        self.orchestrator = SmartMentorOrchestrator(config)    
        self.rouge = RougeEval()    
        logger.info("Evaluation Models is on!")
    
    def get_prompt(self, question, hypothesis, **kwargs):
        return self.orchestrator.prepare_prompt(question, hypothesis, **kwargs)

    def get_response_openai_by_prompt(self, prompt):
        return self.orchestrator.request_openai_by_prompt(prompt)

    def get_response_llama_by_prompt(self, prompt):
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
        
    def get_response(self, response:str):

        string_search = '"program_created":'
        index_string = response.find(string_search) + len(string_search)
        new_response = response[index_string:]

        if new_response.find('"""') == -1:
            new_response = new_response.replace('"','',1)
            if new_response.rfind('}') != -1:
                index_string = new_response.rfind('}') - 2
            else:
                index_string = new_response.rfind(']') - 2
            new_response = new_response[:index_string]
        else:
            new_response = new_response.replace('"""','',1)
            if new_response.rfind('}') != -1:
                index_string = new_response.rfind('}') - 2
            else:
                index_string = new_response.rfind(']') - 2
            new_response = new_response[:index_string]

        return new_response
    
    @property
    def generate_random_numbers(self):
        random_numbers = [random.randint(0, 70) for _ in range(1)]
        return pd.DataFrame({'index': random_numbers})


if __name__ == "__main__":
    config = ConfigHelper()
    reader = SmartReader()
    writer = SmartWriter()
    models = EvaluationModels(config)

    ## Creating file
    file_random = "smart_mentor/resources/random_numbers.csv"
    df_indexes = pd.DataFrame()
    if not reader.checkFile(file_random):
        writer.write(file_random, models.generate_random_numbers)
        df_indexes = reader.readFile(file_random)
    else:
        df_indexes = reader.readFile(file_random)

    df = reader.readFile("smart_mentor/resources/ground_truth_data.csv")
    df_selected = df.iloc[df_indexes['index']]
    df_metrics = pd.DataFrame(columns=models.COLUMNS_METRICS)

    hypothesis = "h0"
    file_written = f"smart_mentor/resources/Metrics_{hypothesis}.csv"
    ## Creating file
    try:
        reader.removeFile(file_written)
    except FileNotFoundError as e:
        logger.error("File does not found")
    
    writer.write(file_written, df_metrics)

    # how many times I'm going to call the models
    for count in range(2):
        logger.info(f"###### Counting {count} times ######")
        df_metrics = pd.DataFrame(columns=models.COLUMNS_METRICS)
        for row in df_selected.itertuples(): 
            user_question = ""
            if len(str(row._9).replace("Dicas&Dicas","")) > 0:
                user_question = f"{row._2} \n {row._5} \n {row._6} \n {row._7} \n # Dica sobre a questão {row._9}"
            else:
                user_question = f"{row._2} \n {row._5} \n {row._6} \n {row._7}"
            reference = row._20

            if hypothesis == "h11":
                logger.info("#### OPENAI")
                new_prompt = models.get_prompt(hypothesis=hypothesis, question=user_question)            
                response = "" 
                for i in range(1,3):                
                    response = models.get_response_openai_by_prompt(prompt=new_prompt)
                    thought = i+1
                    new_response = f'{user_question} \n {response}'
                    new_prompt = models.get_prompt(hypothesis=hypothesis, question=new_response,thought=thought)
                    time.sleep(60)
                
                logger.info(f"#### OPENAI response \n {response}") 
                json_data = json.loads(response)
                list_metrics = models.get_metrics(hypothesis=hypothesis,
                                                  model="openai",
                                                  orig_data=reference,
                                                  predict=json_data["program_created"])
                df_metrics = models.add_new_row(df_metrics, list_metrics)

                logger.info("#### LLAMA")
                new_prompt = models.get_prompt(hypothesis=hypothesis, question=user_question)
                response = "" 
                for i in range(1,3):               
                    response = models.get_response_llama_by_prompt(prompt=new_prompt)
                    thought = i+1
                    new_response = f'{user_question} \n {response}'
                    new_prompt = models.get_prompt(hypothesis=hypothesis, question=new_response,thought=thought)
                    time.sleep(60)

                logger.info(f"#### LLAMA response \n {response}") 
                try:
                    json_data = json.loads(response)
                    response = json_data["program_created"]
                except Exception as e:
                    response = models.get_response(response)

                list_metrics = models.get_metrics(hypothesis=hypothesis,
                                                  model="llama", 
                                                  orig_data=reference,
                                                  predict=response)
                df_metrics = models.add_new_row(df_metrics, list_metrics)
            else:
                ### Others hypotheses
                new_prompt = models.get_prompt(hypothesis=hypothesis, question=user_question)
                response = models.get_response_openai_by_prompt(prompt=new_prompt)
                print(f"#### OPENAI \n {response}") 
                json_data = json.loads(response)
                list_metrics = models.get_metrics(hypothesis=hypothesis,
                                                  model="openai",
                                                  orig_data=reference,
                                                  predict=json_data["program_created"])
                for metrics in list_metrics:
                    logger.info(f'From {metrics["metric"]} by rouge_score library -> Precision: {metrics["precision"]} Recall: {metrics["recall"]} fmeasure: {metrics["f1_score"]} ')

                response = models.get_response_llama_by_prompt(prompt=new_prompt)
                print(f"#### LLAMA \n {response}") 
                try:
                    json_data = json.loads(response)
                    response = json_data["program_created"]
                except Exception as e:
                    response = models.get_response(response)

                list_metrics = models.get_metrics(hypothesis=hypothesis,
                                                  model="llama", 
                                                  orig_data=reference,
                                                  predict=response)
                df_metrics = models.add_new_row(df_metrics, list_metrics)
            
            logger.info(f'Finished the index {row.Index}')
            time.sleep(60)

        ## Appending dataframe
        writer.write(file_written, df_metrics)            