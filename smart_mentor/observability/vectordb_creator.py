from ..config.config_helper import ConfigHelper
from ..rags.retriever import RAG
from ..file.smart_reader import SmartReader
from ..config import logging_config
import pandas as pd

logger = logging_config.setup_logging()

if __name__ == "__main__":
    config = ConfigHelper()
    reader = SmartReader()
    df = pd.DataFrame()
    embedding_config = {"embedding_version": config.get_config.AZURE_OPENAI_API_VERSION,
                        "openai_key": config.get_config.AZURE_OPENAI_API_KEY,
                        "openai_endpoint": config.get_config.AZURE_OPENAI_ENDPOINT,
                        }
    
    db_dir = f'smart_mentor/database/vectordb/{config.get_config.AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDING}/chroma_db'
    
    rag = RAG(embedding_config=embedding_config, 
              name_deployment=config.get_config.AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDING,
              persist_dir=db_dir,
              top_k=3)
    
    if reader.checkFile("smart_mentor/resources/ground_truth_data.csv"):
        df = reader.readFile("smart_mentor/resources/ground_truth_data.csv")
        
    # Data columns (total 21 columns):
    #  #   Column                                Non-Null Count  Dtype  
    # ---  ------                                --------------  -----  
    #  0   model_x                               100 non-null    object 
    #  1   pk_x                                  100 non-null    object 
    #  2   fields.title_x                        100 non-null    object 
    #  3   fields.code_x                         100 non-null    object 
    #  4   fields.judge_x                        100 non-null    object 
    #  5   fields.desc_x                         100 non-null    object 
    #  6   fields.input_desc_x                   100 non-null    object 
    #  7   fields.output_desc_x                  100 non-null    object 
    #  8   fields.examples_desc_x                100 non-null    object 
    #  9   fields.dicas_x                        100 non-null    object 
    #  10  fields.origin_x                       100 non-null    object 
    #  11  fields.tags_x                         100 non-null    object 
    #  12  model_y                               100 non-null    object 
    #  13  pk_y                                  100 non-null    object 
    #  14  fields.user_y                         100 non-null    float64
    #  15  fields.problem_y                      100 non-null    float64
    #  16  fields.code_y                         100 non-null    object 
    #  17  fields.language_y                     100 non-null    object 
    #  18  fields.judge_result.judge_veredict_y  100 non-null    object 
    #  19  fields.problem_rating_delta_y         100 non-null    float64
    #  20  fields.program_y                      100 non-null    objec

    for row in df.itertuples(index=False):        
        text_question = f"{row._2} \n {row._5} \n {row._6} \n {row._7} \n # Dica sobre a questão {row._9} \n Resposta: \n {row._20}"
        # text_answer = f"{row._20}"
        rag.save_documents(question=text_question)
    
    prompt = "Escreva o mundo é dos homens"  
    docs = rag.retrieve(prompt) 
    logger.info(f"Retrieved docs: {docs}")
    logger.info(f"Metadatas: {rag.retrieve_metadata()}")
    docs_question = rag.retrieve(prompt,['question']) 
    logger.info(f"Retrieved docs: {docs_question}")
    for doc in docs_question:
        logger.info(f" Question and Answer: {doc.page_content}")
    # docs_answer = rag.retrieve(prompt,['answer']) 
    # logger.info(f"Retrieved docs: {docs_answer}")