from .prompt_role import PromptRole
from .prompt_rar import PromptRaR
from .prompt_zero_cot import PromptZeroCoT

from ..config import logging_config

logger = logging_config.setup_logging()

class PromptHandler:    
    def __init__(self, answer: str, rag: str):        
        self.answer = answer        
        self.rag = rag
        self.role = PromptRole()
        self.rar = PromptRaR()
        self.zcot = PromptZeroCoT()        

    def generatePromptH8(self):
        return self.get_messages(user_content=self.answer)

    def generatePromptH9(self):
        user_content = f'{self.answer} \n {self.rar.rar}'
        return self.get_messages(user_content=user_content)
    
    def generatePromptH10(self):
        user_content = f'{self.answer} \n {self.zcot.zero_cot_opt3}'
        return self.get_messages_no_rag(user_content=user_content)

    def get_messages(self, user_content: str):
        messages=[
            {"role": "system",
            "content": self.role.string_role_complete},
            {"role":"system", 
            "content": self.get_rag()},
            {"role": "user", 
            "content": user_content}]
        
        return messages
    
    def get_messages_no_rag(self, user_content: str):
        messages=[
            {"role": "system",
            "content": self.role.string_role_complete},
            {"role": "user", 
            "content": user_content}]
        
        return messages

    def get_rag(self):
        rag_content =  ""
        for text in self.rag:
            rag_content =  rag_content + f'{text}\n'
        return rag_content