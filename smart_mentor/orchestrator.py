from typing import Final
from .config.config_helper import ConfigHelper
from .modelos_api.openai.client_openAi import ClientOpenAi
from .modelos_api.llamaai.client_llamaAi import ClientLlamaOpenAi
from .prompts.promptHandler import PromptHandler
from .rags.retriever import RAG
from .config import logging_config


logger = logging_config.setup_logging()

class SmartMentorOrchestrator:

    TOP_K: Final = 3

    def __init__(self, config: ConfigHelper):           

        openai_config = {"client_version": config.get_config.AZURE_OPENAI_API_VERSION,
                         "key": config.get_config.AZURE_OPENAI_API_KEY,
                         "endpoint": config.get_config.AZURE_OPENAI_ENDPOINT,
                         "model": config.get_config.AZURE_OPENAI_DEPLOYMENT_NAME                        
                        }
        
        claudeai_config = {"key": config.get_config.CLAUDEAI_KEY,                           
                           "model": config.get_config.CLAUDEAI_MODEL                        
                          }
        
        llama_config = {"key": config.get_config.LLAMA_API_KEY,
                        "endpoint": config.get_config.LLAMA_ENDPOINT,
                        "model": config.get_config.LLAMA_MODEL
                       }
        
        embedding_config = {"embedding_version": config.get_config.AZURE_OPENAI_API_VERSION,
                            "key": config.get_config.AZURE_OPENAI_API_KEY,
                            "endpoint": config.get_config.AZURE_OPENAI_ENDPOINT,
                            }
        
        self.db_dir = f'smart_mentor/database/vectordb/{config.get_config.AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDING}/chroma_db'
     
        self.rag = RAG(embedding_config=embedding_config, 
                       name_deployment=config.get_config.AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDING, 
                       persist_dir=self.db_dir, 
                       top_k=self.TOP_K) 
        self.clientOpenAI = ClientOpenAi(openai_config)
        self.clientLlamaAI = ClientLlamaOpenAi(llama_config)

    def get_rag(self, question:str):
        logger.info("Researching vectordb")
        docs = self.rag.retrieve(question)
        return docs
        
    def prepare_prompt(self, question:str, hypothesis: str, model:str, **kwargs):
        thought = kwargs.get("thought", 1)
        first_step = kwargs.get("first_step", True)
        docs = []
        match hypothesis:
            case 'h5' | 'h9':
                if thought == 3:
                    logger.info("Calling rag")
                    docs = self.get_rag(question)
            case 'h8' | 'h10' | 'h11':
                if first_step:
                    logger.info("Calling rag")
                    docs = self.get_rag(question)
            case _:
                if hypothesis != 'h6_conclusions':
                    logger.info("Calling rag")
                    docs = self.get_rag(question)

        logger.info("Calling generatePrompt")
        assistant = False if model == "openai" else True
        prompt = PromptHandler(question, docs, assistant=assistant).generatePrompt(hypothesis, **kwargs)
        return prompt 

    def request_openai_by_prompt(self, prompt: list):
        logger.info("Requesting OpenAI")
        response = self.clientOpenAI.send_request(
            messages=prompt,
        )        
        return response
    
    def request_llama_by_prompt(self, prompt: list):
        logger.info("Requesting Llama")
        response = self.clientLlamaAI.send_request(
            messages=prompt,
        )        
        return response

