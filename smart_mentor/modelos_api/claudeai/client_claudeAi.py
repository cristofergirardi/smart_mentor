from ...lib.core import ModelAIFactory
from ...config import logging_config
import anthropic

logger = logging_config.setup_logging()

class ClientClaudeAi(ModelAIFactory):

    def __init__(self, config: dict):
        self.model = anthropic.Anthropic(api_key=config["key"])
        self.model_name = config["model"]        
    
    def send_request(self, **kwargs):
        logger.info("Creating a model response for the given chat conversation...")
        messages = kwargs.get("messages",[])
        max_tokens = kwargs.get("max_tokens",1000)
        temperature = kwargs.get("temperature",0)
        response = self.model.messages.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )        
        return response.content[0].text
