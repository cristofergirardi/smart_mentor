from ...lib.core import ModelAIFactory
from ...config import logging_config
from openai import AzureOpenAI

logger = logging_config.setup_logging()

class ClientOpenAi(ModelAIFactory):

    def __init__(self, config: dict):
        self.model = AzureOpenAI(
            api_version=config["client_version"],
            api_key= config["key"], 
            azure_endpoint = config["endpoint"],
        )
        self.model_name = config["model"]
    
    def send_request(self, **kwargs):
        logger.info("Creating a model response for the given chat conversation...")
        messages = kwargs.get("messages",[])
        max_tokens = kwargs.get("max_tokens",1000)
        temperature = kwargs.get("temperature",0)
        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        try:
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in the model response: {e}")
            return "Sorry, I am not able to answer that question."
