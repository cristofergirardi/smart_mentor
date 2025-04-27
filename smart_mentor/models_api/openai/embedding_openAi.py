from ...config import logging_config
from ...lib.core import ModelAIFactory
from langchain_openai import AzureOpenAIEmbeddings

logger = logging_config.setup_logging()

class EmbeddingOpenAi(ModelAIFactory):

    def __init__(self, config: dict, **kwargs):
        self.api_version = config["embedding_version"]
        self.api_key = config["key"] 
        self.azure_endpoint = config["endpoint"]
        self.embedding = AzureOpenAIEmbeddings(
                    azure_deployment = kwargs.get("name_deployment","text-embedding-3"),
                    openai_api_version = self.api_version,
                    api_key = self.api_key,  
                    azure_endpoint = self.azure_endpoint
                )
    
    def send_request(self, **kwargs):
        logger.info("Sending an embedding...")
        return self.embedding
