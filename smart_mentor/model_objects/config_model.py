from pydantic import BaseModel

class ConfigModel(BaseModel):

    AZURE_OPENAI_API_VERSION: str
    AZURE_OPENAI_DEPLOYMENT_NAME: str
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDING: str    
    CLAUDEAI_KEY: str
    CLAUDEAI_MODEL: str
    LLAMA_API_KEY: str
    LLAMA_ENDPOINT: str
    LLAMA_MODEL: str
    QWEN_MODEL: str