from typing import Final
from .config.config_helper import ConfigHelper
from .modelos_api.openai.client_openAi import ClientOpenAi
from .prompts.promptHandler import PromptHandler
from .rags.retriever import RAG
from .config import logging_config

logger = logging_config.setup_logging()

class SmartMentorOrchestrator:

    TOP_K: Final = 5

    def __init__(self, config: ConfigHelper):           

        client_config = {"client_version": config.get_config.AZURE_OPENAI_API_VERSION,
                         "openai_key": config.get_config.AZURE_OPENAI_API_KEY,
                         "openai_endpoint": config.get_config.AZURE_OPENAI_ENDPOINT,
                         "model": config.get_config.AZURE_OPENAI_DEPLOYMENT_NAME                        
                        }
        
        embedding_config = {"embedding_version": config.get_config.AZURE_OPENAI_API_VERSION,
                            "openai_key": config.get_config.AZURE_OPENAI_API_KEY,
                            "openai_endpoint": config.get_config.AZURE_OPENAI_ENDPOINT,
                            }
        
        self.db_dir = f'smart_mentor/database/vectordb/{config.get_config.AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDING}/chroma_db'
     
        self.rag = RAG(embedding_config=embedding_config, 
                       name_deployment=config.get_config.AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDING, 
                       persist_dir=self.db_dir, 
                       top_k=self.TOP_K) 
        self.clientOpenAI = ClientOpenAi(client_config)

    def request(self, question):
        docs = self.rag.retrieve(question) 
        prompt = PromptHandler(question, docs).generatePrompt()
        response = self.clientOpenAI.send_request(
            messages=prompt,
        )
        
        return response.choices[0].message.content
