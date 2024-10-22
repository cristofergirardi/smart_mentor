from pydantic import BaseModel

class ConfigModel(BaseModel):
    openai_key : str
    llama_key : str
    claude_key : str