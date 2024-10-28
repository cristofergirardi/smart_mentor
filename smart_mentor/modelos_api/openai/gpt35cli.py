from langchain_openai import ChatOpenAI
from ...model_objects.config_model import ConfigModel

class GPT35Cli():

    def __init__(self) -> None:
        pass

    def get_llm(config: ConfigModel):
        llm = ChatOpenAI(
            model="gpt-3.5",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key= config.openai_key            
        )