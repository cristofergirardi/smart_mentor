from ..lib.core import PromptEng
from .prompt_role import PromptRole
from .prompt_rar import PromptRaR
from .prompt_zero_cot import PromptZeroCoT

from ..config import logging_config

logger = logging_config.setup_logging()

class PromptHandler(PromptEng):

    def __init__(self, answer: str, rag: str, assistant: bool = False):        
        self.answer = answer        
        self.rag = rag
        self.assistant = assistant
        self.role = PromptRole()
        self.rar = PromptRaR()
        self.zcot = PromptZeroCoT()

    def _generatePromptH8(self):
        logger.info("Calling Hypotheses 8")
        system_content = f'{self.role.string_role_complete}' 
        return self.prompt_message(system_content=system_content,
                                   user_content=self.answer,
                                   role_type= "assistant" if self.assistant else "system")         

    def _generatePromptH9(self):
        logger.info("Calling Hypotheses 9")
        system_content = f'{self.role.string_role_complete}' 
        user_content = f'{self.answer} \n {self.rar.rar}'
        return self.prompt_message(system_content=system_content,
                                   user_content=user_content,
                                   role_type= "assistant" if self.assistant else "system")   

    def _generatePromptH10(self):
        logger.info("Calling Hypotheses 10")
        system_content = f'{self.role.string_role_complete}' 
        user_content = f'{self.answer} \n {self.zcot.zero_cot_opt1}'
        return self.prompt_message(system_content=system_content,
                                   user_content=user_content,
                                   role_type= "assistant" if self.assistant else "system")

    def prompt_message(self,**kwargs):
        message = []
        system = kwargs.get("system_content", "")
        user_content = kwargs.get("user_content", None)
        role_type = kwargs.get("role_type", "system")

        message.append(
            {"role": role_type,
            "content": system}
            )

        message.append(
            {"role": role_type,
            "content": self._get_rag()}
            )
            
        if user_content is not None:
            message.append(
                {"role": "user",
                "content": user_content}
                )
        return message

    def _get_rag(self):
        rag_content =  ""
        for text in self.rag:
            rag_content =  rag_content + f'{text}\n'
        return rag_content
    
    def generatePrompt(self, hypotheses:str):        
        match hypotheses:
            case "h8":
                return self._generatePromptH8()
            case "h9":
                return self._generatePromptH9()
            case "h10":
                return self._generatePromptH10()
            case _:
                return "Unknown hypotheses choise again."
