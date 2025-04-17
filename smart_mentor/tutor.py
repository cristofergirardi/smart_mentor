from .config.config_helper import ConfigHelper
from .orchestrator import SmartMentorOrchestrator
from .config import logging_config
from .observability.rouge_eval import RougeEval
from .observability.bert_similarity import BertSimilarity
from .observability.codet5_similarity import CodeT5Similarity
from .file.smart_reader import SmartReader
from .file.smart_writer import SmartWriter
from typing import Final
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import random
import json
import time

logger = logging_config.setup_logging()

class SmartMentor():

    ROUGE_METRICS: Final = ['rouge1', 'rouge2', 'rougeL']
    COLUMNS_METRICS_ROUGE: Final = ["h", "model", "metric", "precision", "recall", "f1_score"]
    COLUMNS_METRICS_BERT: Final = ["h", "model", "metric", "similarity"]

    def __init__(self, config: ConfigHelper):
        super().__init__()
        self.orchestrator = SmartMentorOrchestrator(config)      
        self.rouge = RougeEval()
        self.bert_similar = BertSimilarity()
        self.codet5_similar = CodeT5Similarity()
        logger.info("Smart Tutor is on!")
    
    def get_prompt(self, question:str, hypothesis:str, model:str, **kwargs):
        return self.orchestrator.prepare_prompt(question, hypothesis, model, **kwargs)
            
    def get_response_openai_by_prompt(self, prompt):
        return self.orchestrator.request_openai_by_prompt(prompt)

    def get_response_llama_by_prompt(self, prompt):
        return self.orchestrator.request_llama_by_prompt(prompt)
    
    def get_metrics_rouge(self, hypothesis: str, model:str, orig_data: str, predict: str) -> list:        
        self.rouge.get_scores(reference=orig_data, result=predict)
        metrics_list = []
        for metric in self.ROUGE_METRICS:
            precision, recall, fmeasure = self.rouge.get_score_by_name(metric_name=metric)
            metrics_list.append(
                {"h": hypothesis, "model": model, "metric": metric, "precision": precision, "recall": recall, "f1_score": fmeasure}
            )
        return metrics_list
    
    def get_metrics_bert(self, hypothesis: str, model:str, orig_data: str, predict: str) -> list:
        metrics_list = []
        similarity = self.bert_similar.get_similarity(ground_truth=orig_data,
                                                      result=predict)
        metrics_list.append(
                {"h": hypothesis, "model": model, "metric": "bert_metric", "similarity": similarity}
            )        
        return metrics_list
    
    def get_metrics_codet5(self, hypothesis: str, model:str, orig_data: str, predict: str) -> list:
        metrics_list = []
        similarity = self.codet5_similar.get_similarity(ground_truth=orig_data, 
                                                        result=predict)
        metrics_list.append(
                {"h": hypothesis, "model": model, "metric": "codet5_metric", "similarity": similarity}
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

    def get_metrics_overall(self, hypothesis: str, model:str, response: str, reference:str):
        new_response = self.extract_programa_gen(response)

        list_metrics_rouge = self.get_metrics_rouge(hypothesis=hypothesis,
                                                    model=model, 
                                                    orig_data=reference,
                                                    predict=new_response)

        list_metrics_bert = self.get_metrics_bert(hypothesis=hypothesis,
                                                  model=model, 
                                                  orig_data=reference,
                                                  predict=new_response)
        
        list_metrics_codet5 = self.get_metrics_codet5(hypothesis=hypothesis,
                                                      model=model, 
                                                      orig_data=reference,
                                                      predict=new_response)

        return list_metrics_rouge, list_metrics_bert, list_metrics_codet5
     
    def show_metrics(self, list_metrics_rouge: list, list_metrics_bert: list, list_metrics_codet5: list):
        for metrics in list_metrics_rouge:
            logger.info(f'From {metrics["metric"]} by rouge_score library -> Precision: {metrics["precision"]} Recall: {metrics["recall"]} fmeasure: {metrics["f1_score"]} ')
                
        for metrics in list_metrics_bert:
            logger.info(f'From {metrics["metric"]} library -> Similarity: {metrics["similarity"]}')
        
        for metrics in list_metrics_codet5:
            logger.info(f'From {metrics["metric"]} library -> Similarity: {metrics["similarity"]}')
    
    def extract_programa_gen(self, response:str):
        new_response = ""
        try:
            json_data = json.loads(response)
            new_response = json_data["program_created"]
        except Exception as e:
            if type(response) == list:
                new_response = self.get_response(response[0])
            else:
                new_response = self.get_response(response)

        # Check if the new_response is a list
        if type(new_response) == list:
            return ' '.join(new_response)
        else: 
            return new_response
        
    def get_list_parameters(self, hypothesis: str, question:str, model:str):
        list_param = []
        for thought in range(1,4):
            list_param.append({
                "hypothesis": hypothesis,
                "question": question,
                "model": model,
                "thought": thought,
                "fisrt_step": True if thought == 1 else False
            })
        return list_param
    
    def run_parallel_skeleton(self, df_param):
        new_prompt = self.get_prompt(hypothesis = df_param["hypothesis"], 
                                     question = df_param["question"], 
                                     model = df_param["model"],
                                     thought = df_param["thought"],
                                     fisrt_step = df_param["fisrt_step"])  
        response = ""
        if df_param["model"] == "openai":
            response = self.get_response_openai_by_prompt(prompt=new_prompt)
        else:
            response = self.get_response_llama_by_prompt(prompt=new_prompt)
        return response

if __name__ == "__main__":
    config = ConfigHelper()
    reader = SmartReader()
    writer = SmartWriter()
    tutor = SmartMentor(config)
    hypothesis = "h0"

    ## Creating file
    file_random = "smart_mentor/resources/random_numbers.csv"
    df_indexes = pd.DataFrame()
    if not reader.checkFile(file_random):
        writer.write(file_random, tutor.generate_random_numbers)
        df_indexes = reader.readFile(file_random)
    else:
        df_indexes = reader.readFile(file_random)

    df = reader.readFile("smart_mentor/resources/ground_truth_data.csv")
    df.columns = df.columns.str.replace('.', '_')
    df_selected = df.iloc[df_indexes['index']]    

    for row in df_selected.itertuples(index=False): 
        prompt = ""
        if len(str(row.fields_dicas_x).replace("Dicas&Dicas","")) > 0:
            prompt = f'''Title of issue: {row.fields_title_x} \n 
                        Activity description: {row.fields_desc_x} \n 
                        Input Data: {row.fields_input_desc_x} \n 
                        Output Data: {row.fields_output_desc_x} \n 
                        Question clue: {row.fields_dicas_x}
                    '''
        else:
            prompt = f'''Title of issue: {row.fields_title_x} \n 
                        Activity description: {row.fields_desc_x} \n 
                        Input Data: {row.fields_input_desc_x} \n 
                        Output Data: {row.fields_output_desc_x} 
                    '''
        reference = row.fields_program_y

        match hypothesis:
            case "h0":
                logger.info("#### OPENAI")
                new_prompt = tutor.get_prompt(hypothesis=hypothesis, question=prompt, model="openai")
                response = tutor.get_response_openai_by_prompt(prompt=new_prompt)
                time.sleep(60)
                logger.info(f"#### OPENAI response \n {response}") 
                list_metrics_rouge, list_metrics_bert, list_metrics_codet5 = tutor.get_metrics_overall(hypothesis=hypothesis,
                                                                                                       model="openai", 
                                                                                                       reference=reference, 
                                                                                                       response=response)

                tutor.show_metrics(list_metrics_rouge, 
                                   list_metrics_bert, 
                                   list_metrics_codet5)

                logger.info("#### LLAMA")
                new_prompt = tutor.get_prompt(hypothesis=hypothesis, question=prompt, model="llama")
                response = tutor.get_response_llama_by_prompt(prompt=new_prompt)
                time.sleep(60)
                logger.info(f"#### LLAMA response \n {response}") 
                list_metrics_rouge, list_metrics_bert, list_metrics_codet5 = tutor.get_metrics_overall(hypothesis=hypothesis,
                                                                                                       model="llama", 
                                                                                                       reference=reference, 
                                                                                                       response=response)
                tutor.show_metrics(list_metrics_rouge, 
                                   list_metrics_bert, 
                                   list_metrics_codet5)
            case "h5" | "h9":
                logger.info("#### OPENAI")
                list_parameters = tutor.get_list_parameters(hypothesis=hypothesis, question=prompt, model="openai")
                results = []
                with ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(tutor.run_parallel_skeleton, parameters)
                        for parameters in list_parameters
                    ]
                    
                    for future in futures:
                        results.append(future.result())

                new_response = " ".join(results)
                time.sleep(60)
                new_prompt = tutor.get_prompt(hypothesis=hypothesis, question=new_response, model="openai",thought=4)
                response = tutor.get_response_openai_by_prompt(prompt=new_prompt)
                time.sleep(60)
             
                logger.info(f"#### OPENAI response \n {response}") 
                list_metrics_rouge, list_metrics_bert, list_metrics_codet5 = tutor.get_metrics_overall(hypothesis=hypothesis,
                                                                                                       model="openai", 
                                                                                                       reference=reference, 
                                                                                                       response=response)

                tutor.show_metrics(list_metrics_rouge, 
                                   list_metrics_bert, 
                                   list_metrics_codet5)

                logger.info("#### LLAMA")
                list_parameters = tutor.get_list_parameters(hypothesis=hypothesis, question=prompt, model="llama")
                results = []
                with ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(tutor.run_parallel_skeleton, parameters)
                        for parameters in list_parameters
                    ]
                    
                    for future in futures:
                        results.append(future.result())

                new_response = " ".join(results)
                time.sleep(60)
                new_prompt = tutor.get_prompt(hypothesis=hypothesis, question=new_response, model="llama",thought=4)
                response = tutor.get_response_llama_by_prompt(prompt=new_prompt)
                time.sleep(60)

                logger.info(f"#### LLAMA response \n {response}") 
                list_metrics_rouge, list_metrics_bert, list_metrics_codet5 = tutor.get_metrics_overall(hypothesis=hypothesis,
                                                                                                       model="llama", 
                                                                                                       reference=reference, 
                                                                                                       response=response)
                tutor.show_metrics(list_metrics_rouge, 
                                   list_metrics_bert, 
                                   list_metrics_codet5)
                
            case "h6":
                logger.info("#### OPENAI")
                new_prompt = tutor.get_prompt(hypothesis=hypothesis, question=prompt, model="openai")
                response = tutor.get_response_openai_by_prompt(prompt=new_prompt)  
                time.sleep(60)        
                list_response = []
                for i in range(0,3):    
                    hypothesis = 'h6_conclusions'           
                    new_response = f'{prompt} \n Response: {tutor.extract_programa_gen(response)}'
                    new_prompt = tutor.get_prompt(hypothesis=hypothesis, question=new_response, model="openai")
                    logger.info(f"#### Prompt \n {new_prompt}")
                    response = tutor.get_response_openai_by_prompt(prompt=new_prompt) 
                    list_metrics_rouge, list_metrics_bert, list_metrics_codet5 = tutor.get_metrics_overall(hypothesis=hypothesis, 
                                                                                                           model="openai", 
                                                                                                           reference=reference, 
                                                                                                           response=response)
                    tutor.show_metrics(list_metrics_rouge,
                                       list_metrics_bert, 
                                       list_metrics_codet5)
                    
                    list_response.append({
                        "response": response,
                        "metric": list_metrics_bert[0]["similarity"]
                    })
                    time.sleep(60)
                
                highest_response = max(list_response, key=lambda item: item.get("metric",0))
                logger.info(f"#### OPENAI response \n {highest_response.get("response","")} \n metric {highest_response.get("metric",0)}") 

                logger.info("#### LLAMA")
                hypothesis = 'h6'
                new_prompt = tutor.get_prompt(hypothesis=hypothesis, question=prompt, model="llama")
                response = tutor.get_response_llama_by_prompt(prompt=new_prompt)
                time.sleep(60)
                list_response = []
                for i in range(0,3): 
                    hypothesis = 'h6_conclusions'           
                    new_response = f'{prompt} \n Response: {tutor.extract_programa_gen(response)}'
                    new_prompt = tutor.get_prompt(hypothesis=hypothesis, question=new_response, model="llama")
                    logger.info(f"#### Prompt \n {new_prompt}")
                    response = tutor.get_response_llama_by_prompt(prompt=new_prompt)
                    list_metrics_rouge, list_metrics_bert, list_metrics_codet5 = tutor.get_metrics_overall(hypothesis=hypothesis, 
                                                                                                           model="llama", 
                                                                                                           reference=reference, 
                                                                                                           response=response)
                    tutor.show_metrics(list_metrics_rouge,
                                       list_metrics_bert, 
                                       list_metrics_codet5)
                    
                    list_response.append({
                        "response": response,
                        "metric": list_metrics_bert[0]["similarity"]
                    })
                    time.sleep(60)
                
                highest_response = max(list_response, key=lambda item: item.get("metric",0))
                logger.info(f"#### LLAMA response \n {highest_response.get("response","")} \n metric {highest_response.get("metric",0)}") 
            
            case "h7":
                logger.info("#### OPENAI")
                new_prompt = tutor.get_prompt(hypothesis=hypothesis, question=prompt, model="openai",thought=0)  
                response = tutor.get_response_openai_by_prompt(prompt=new_prompt)
                time.sleep(60)

                list_parameters = tutor.get_list_parameters(hypothesis=hypothesis, question=response, model="openai")
                results = []
                
                with ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(tutor.run_parallel_skeleton, parameters)
                        for parameters in list_parameters
                    ]
                    
                    for future in futures:
                        results.append(future.result())

                new_response = " ".join(results)
                time.sleep(60)
                new_prompt = tutor.get_prompt(hypothesis=hypothesis, question=new_response, model="openai",thought=4)
                response = tutor.get_response_openai_by_prompt(prompt=new_prompt)
                time.sleep(60)
                
                logger.info(f"#### OPENAI response \n {response}") 
                list_metrics_rouge, list_metrics_bert, list_metrics_codet5 = tutor.get_metrics_overall(hypothesis=hypothesis,
                                                                                                       model="openai", 
                                                                                                       reference=reference, 
                                                                                                       response=response)

                tutor.show_metrics(list_metrics_rouge, 
                                   list_metrics_bert, 
                                   list_metrics_codet5)

                logger.info("#### LLAMA")
                new_prompt = tutor.get_prompt(hypothesis=hypothesis, question=prompt, model="llama",thought=0)  
                response = tutor.get_response_openai_by_prompt(prompt=new_prompt)
                time.sleep(60)

                list_parameters = tutor.get_list_parameters(hypothesis=hypothesis, question=response, model="llama")
                results = []
                with ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(tutor.run_parallel_skeleton, parameters)
                        for parameters in list_parameters
                    ]
                    
                    for future in futures:
                        results.append(future.result())

                new_response = " ".join(results)
                time.sleep(60)
                new_prompt = tutor.get_prompt(hypothesis=hypothesis, question=new_response, model="llama",thought=4)
                response = tutor.get_response_llama_by_prompt(prompt=new_prompt)
                time.sleep(60)

                logger.info(f"#### LLAMA response \n {response}") 
                list_metrics_rouge, list_metrics_bert, list_metrics_codet5 = tutor.get_metrics_overall(hypothesis=hypothesis,
                                                                                                       model="llama", 
                                                                                                       reference=reference, 
                                                                                                       response=response)
                tutor.show_metrics(list_metrics_rouge, 
                                   list_metrics_bert, 
                                   list_metrics_codet5)
    
            case "h8" | 'h10':
                logger.info("#### OPENAI")
                new_prompt = tutor.get_prompt(hypothesis=hypothesis, question=prompt, model="openai")  
                response = tutor.get_response_openai_by_prompt(prompt=new_prompt)
                time.sleep(60)        
                list_response = []
                for i in range(0,3):    
                    new_response = f'{prompt} \n Response: {tutor.extract_programa_gen(response)}'
                    new_prompt = tutor.get_prompt(hypothesis=hypothesis, question=new_response, model="openai", first_step=False)
                    response = tutor.get_response_openai_by_prompt(prompt=new_prompt) 
                    list_metrics_rouge, list_metrics_bert, list_metrics_codet5 = tutor.get_metrics_overall(hypothesis=hypothesis, 
                                                                                                           model="openai", 
                                                                                                           reference=reference, 
                                                                                                           response=response)
                    tutor.show_metrics(list_metrics_rouge,
                                       list_metrics_bert, 
                                       list_metrics_codet5)
                    
                    list_response.append({
                        "response": response,
                        "metric": list_metrics_bert[0]["similarity"]
                    })
                    time.sleep(60)
                
                highest_response = max(list_response, key=lambda item: item.get("metric",0))
                logger.info(f"#### OPENAI response \n {highest_response.get("response","")} \n metric {highest_response.get("metric",0)}") 
                
                logger.info("#### LLAMA")
                new_prompt = tutor.get_prompt(hypothesis=hypothesis, question=prompt, model="llama")  
                response = tutor.get_response_llama_by_prompt(prompt=new_prompt)
                time.sleep(60)        
                list_response = []
                for i in range(0,3):    
                    new_response = f'{prompt} \n Response: {tutor.extract_programa_gen(response)}'
                    new_prompt = tutor.get_prompt(hypothesis=hypothesis, question=new_response, model="llama", first_step=False)
                    response = tutor.get_response_llama_by_prompt(prompt=new_prompt) 
                    list_metrics_rouge, list_metrics_bert, list_metrics_codet5 = tutor.get_metrics_overall(hypothesis=hypothesis, 
                                                                                                           model="llama", 
                                                                                                           reference=reference, 
                                                                                                           response=response)
                    tutor.show_metrics(list_metrics_rouge,
                                       list_metrics_bert, 
                                       list_metrics_codet5)
                    
                    list_response.append({
                        "response": response,
                        "metric": list_metrics_bert[0]["similarity"]
                    })
                    time.sleep(60)
                
                highest_response = max(list_response, key=lambda item: item.get("metric",0))
                logger.info(f"#### LLAMA response \n {highest_response.get("response","")} \n metric {highest_response.get("metric",0)}")

            case 'h11':
                logger.info("#### OPENAI")                
                new_prompt = tutor.get_prompt(hypothesis=hypothesis, question=prompt, model="openai",thought=0)  
                response = tutor.get_response_openai_by_prompt(prompt=new_prompt)
                time.sleep(60)

                list_parameters = tutor.get_list_parameters(hypothesis=hypothesis, question=response, model="openai")
                results = []
                list_response = []
                with ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(tutor.run_parallel_skeleton, parameters)
                        for parameters in list_parameters
                    ]
                    
                    for future in futures:
                        results.append(future.result())

                new_response = " ".join(results)
                time.sleep(60)
                new_prompt = tutor.get_prompt(hypothesis=hypothesis, question=new_response, model="openai",thought=4)
                response = tutor.get_response_openai_by_prompt(prompt=new_prompt)
                time.sleep(60)

                for j in range(0,3):
                    new_response = f'{new_prompt} \n Response: {tutor.extract_programa_gen(response)}'
                    new_prompt = tutor.get_prompt(hypothesis=hypothesis, question=new_response, model="openai", thought=5, first_step=False)
                    response = tutor.get_response_openai_by_prompt(prompt=new_prompt) 
                    list_metrics_rouge, list_metrics_bert, list_metrics_codet5 = tutor.get_metrics_overall(hypothesis=hypothesis, 
                                                                                                            model="openai", 
                                                                                                            reference=reference, 
                                                                                                            response=response)
                    
                    list_response.append({
                        "response": response,
                        "metric": list_metrics_bert[0]["similarity"]
                    })
                    time.sleep(60)

                highest_response = max(list_response, key=lambda item: item.get("metric",0))
                logger.info(f"#### OPENAI response \n {highest_response.get("response","")} \n metric {highest_response.get("metric",0)}")                            

                logger.info("#### LLAMA")
                new_prompt = tutor.get_prompt(hypothesis=hypothesis, question=prompt, model="llama", thought=0)  
                response = tutor.get_response_llama_by_prompt(prompt=new_prompt)
                time.sleep(60)

                list_parameters = tutor.get_list_parameters(hypothesis=hypothesis, question=response, model="llama")
                results = []
                list_response = []
                with ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(tutor.run_parallel_skeleton, parameters)
                        for parameters in list_parameters
                    ]
                    
                    for future in futures:
                        results.append(future.result())

                new_response = " ".join(results)
                time.sleep(60)
                new_prompt = tutor.get_prompt(hypothesis=hypothesis, question=new_response, model="llama",thought=4)
                response = tutor.get_response_openai_by_prompt(prompt=new_prompt)
                time.sleep(60)

                for j in range(0,3):
                    new_response = f'{new_prompt} \n Response: {tutor.extract_programa_gen(response)}'
                    new_prompt = tutor.get_prompt(hypothesis=hypothesis, question=new_response, model="llama", thought=5, first_step=False)
                    response = tutor.get_response_llama_by_prompt(prompt=new_prompt) 
                    list_metrics_rouge, list_metrics_bert, list_metrics_codet5 = tutor.get_metrics_overall(hypothesis=hypothesis, 
                                                                                                            model="llama", 
                                                                                                            reference=reference, 
                                                                                                            response=response)
                    
                    list_response.append({
                        "response": response,
                        "metric": list_metrics_bert[0]["similarity"]
                    })
                    time.sleep(60)

                highest_response = max(list_response, key=lambda item: item.get("metric",0))
                logger.info(f"#### LLAMA response \n {highest_response.get("response","")} \n metric {highest_response.get("metric",0)}")  

            case 'h12':
                logger.info("#### OPENAI")
                list_parameters = tutor.get_list_parameters(hypothesis=hypothesis, question=response, model="openai")
                results = []
                list_response = []
                with ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(tutor.run_parallel_skeleton, parameters)
                        for parameters in list_parameters
                    ]
                    
                    for future in futures:
                        results.append(future.result())

                new_response = " ".join(results)
                time.sleep(60)
                new_prompt = tutor.get_prompt(hypothesis=hypothesis, question=new_response, model="openai",thought=4)
                response = tutor.get_response_openai_by_prompt(prompt=new_prompt)
                time.sleep(60)
                
                for j in range(0,3):
                    new_response = f'{new_prompt} \n Response: {tutor.extract_programa_gen(response)}'
                    new_prompt = tutor.get_prompt(hypothesis=hypothesis, question=new_response, model="openai", thought=5, first_step=False)
                    response = tutor.get_response_openai_by_prompt(prompt=new_prompt) 
                    list_metrics_rouge, list_metrics_bert, list_metrics_codet5 = tutor.get_metrics_overall(hypothesis=hypothesis, 
                                                                                                            model="openai", 
                                                                                                            reference=reference, 
                                                                                                            response=response)
                    
                    list_response.append({
                        "response": response,
                        "metric": list_metrics_bert[0]["similarity"]
                    })
                    time.sleep(60)

                highest_response = max(list_response, key=lambda item: item.get("metric",0))
                logger.info(f"#### OPENAI response \n {highest_response.get("response","")} \n metric {highest_response.get("metric",0)}")

                logger.info("#### LLAMA")
                list_parameters = tutor.get_list_parameters(hypothesis=hypothesis, question=response, model="llama")
                results = []
                list_response = []
                with ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(tutor.run_parallel_skeleton, parameters)
                        for parameters in list_parameters
                    ]
                    
                    for future in futures:
                        results.append(future.result())

                new_response = " ".join(results)
                time.sleep(60)
                new_prompt = tutor.get_prompt(hypothesis=hypothesis, question=new_response, model="llama",thought=4)
                response = tutor.get_response_openai_by_prompt(prompt=new_prompt)

                for j in range(0,3):
                    new_response = f'{new_prompt} \n Response: {tutor.extract_programa_gen(response)}'
                    new_prompt = tutor.get_prompt(hypothesis=hypothesis, question=new_response, model="llama", thought=5, first_step=False)
                    response = tutor.get_response_llama_by_prompt(prompt=new_prompt) 
                    list_metrics_rouge, list_metrics_bert, list_metrics_codet5 = tutor.get_metrics_overall(hypothesis=hypothesis, 
                                                                                                            model="llama", 
                                                                                                            reference=reference, 
                                                                                                            response=response)
                    
                    list_response.append({
                        "response": response,
                        "metric": list_metrics_bert[0]["similarity"]
                    })
                    time.sleep(60)

                highest_response = max(list_response, key=lambda item: item.get("metric",0))
                logger.info(f"#### LLAMA response \n {highest_response.get("response","")} \n metric {highest_response.get("metric",0)}") 

            case _:
                ### Others hypotheses
                new_prompt = tutor.get_prompt(hypothesis=hypothesis, question=prompt, model="openai")  
                response = tutor.get_response_openai_by_prompt(prompt=new_prompt)
                print(f"#### OPENAI \n {response}") 
                list_metrics_rouge, list_metrics_bert, list_metrics_codet5 = tutor.get_metrics_overall(hypothesis=hypothesis,
                                                                                                       model="openai", 
                                                                                                       reference=reference, 
                                                                                                       response=response)
                tutor.show_metrics(list_metrics_rouge, 
                                   list_metrics_bert, 
                                   list_metrics_codet5)

                new_prompt = tutor.get_prompt(hypothesis=hypothesis, question=prompt, model="llama")
                response = tutor.get_response_llama_by_prompt(prompt=new_prompt)
                print(f"#### LLAMA \n {response}") 
                list_metrics_rouge, list_metrics_bert, list_metrics_codet5 = tutor.get_metrics_overall(hypothesis=hypothesis,
                                                                                                       model="llama", 
                                                                                                       reference=reference, 
                                                                                                       response=response)
                tutor.show_metrics(list_metrics_rouge, 
                                   list_metrics_bert, 
                                   list_metrics_codet5)

