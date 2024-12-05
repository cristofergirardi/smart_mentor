from ..lib.core import PromptEng
from .prompt_role import PromptRole
from .prompt_rar import PromptRaR
from .prompt_zero_cot import PromptZeroCoT
from .prompt_response import PromptResponse
from .prompt_skeleton_thought import PromptSkeleton
from ..config import logging_config

logger = logging_config.setup_logging()

class PromptHandler(PromptEng):

    def __init__(self, answer: str, rag: str, assistant: bool = False):        
        self.answer = answer        
        self.rag = rag
        self.assistant = assistant
        self.role = PromptRole()
        self.rar = PromptRaR()
        self.prompt_response = PromptResponse()

    def _generatePromptH8(self):
        logger.info("Calling Hypotheses 8")
        system_content = f'{self.role.string_role_complete}' 
        return self.prompt_message(system_content=system_content,
                                   rag_content = self._get_rag(),
                                   user_content=self.answer,
                                   role_type= "assistant" if self.assistant else "system")         

    def _generatePromptH9(self):
        logger.info("Calling Hypotheses 9")
        system_content = f'{self.role.string_role_complete}' 
        user_content = f'{self.answer} \n {self.rar.rar}'
        return self.prompt_message(system_content=system_content,
                                   rag_content = self._get_rag(),
                                   user_content=user_content,
                                   role_type= "assistant" if self.assistant else "system")   

    def _generatePromptH10(self):
        logger.info("Calling Hypotheses 10")
        zcot = PromptZeroCoT()
        system_content = f'{self.role.string_role_complete}' 
        user_content = f'{self.answer} \n {zcot.zero_cot_opt1}'
        return self.prompt_message(system_content=system_content,
                                   rag_content = self._get_rag(),
                                   user_content=user_content,
                                   role_type= "assistant" if self.assistant else "system")
    
    def _generatePromptH11(self, **kwargs):
        logger.info("Calling Hypotheses 11")
        thought = kwargs.get("thought", 1)
        system_content = f'{self.role.string_role_complete}'
        skeleton = PromptSkeleton(question=self.answer)
        user_content = ""
        rag_content = None
        match thought:
            case 1:
                logger.info(f"Creating thought {thought}")
                user_content = f'{self.answer} \n {skeleton.first_think}'                               
            case 2:
                logger.info(f"Creating thought {thought}")
                user_content = f'{self.answer} \n {skeleton.second_think}'
            case 3:
                logger.info(f"Creating thought {thought}")
                rag_content = self._get_rag() 
                user_content = f'{self.answer} \n {skeleton.third_think}'
        return self.prompt_message(system_content=system_content,
                                   rag_content = rag_content,
                                   user_content=user_content,
                                   role_type= "assistant" if self.assistant else "system")

    def prompt_message(self,**kwargs):
        message = []
        system = kwargs.get("system_content", "")
        user_content = kwargs.get("user_content", None)
        rag_content = kwargs.get("rag_content", None)
        role_type = kwargs.get("role_type", "system")

        message.append({
            "role": role_type,
            "content": system
            })

        if rag_content is not None:
            message.append({
                "role": role_type,
                "content": rag_content
                })
            
        message.append({
            "role": role_type,
            "content": self.prompt_response.response
            })

        message.append({
            "role": "user",
            "content": user_content
            })
        return message

    def _get_rag(self):
        rag_content =  ""
        for text in self.rag:
            rag_content =  rag_content + f'{text}\n'
        return rag_content
    
    def generatePrompt(self, hypotheses:str, **kwargs):
        match hypotheses:
            case "h8":
                return self._generatePromptH8()
            case "h9":
                return self._generatePromptH9()
            case "h10":
                return self._generatePromptH10()
            case "h11":
                return self._generatePromptH11(**kwargs)
            case _:
                return "Unknown hypotheses choise again."
