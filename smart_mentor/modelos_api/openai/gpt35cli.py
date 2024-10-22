from langchain_openai import ChatOpenAI

class GPT35Cli():

    def __init__(self) -> None:
        pass

    def get_llm():
        llm = ChatOpenAI(
            model="gpt-3.5",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key="..."            
        )