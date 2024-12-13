from ..lib.core import PromptEng
from .prompt_role import PromptRole
from .prompt_rar import PromptRaR
from .prompt_zero_cot import PromptZeroCoT
from .prompt_response import PromptResponse
from .prompt_skeleton_thought import PromptSkeleton
from .prompt_self_verification import PromptSelfVerification
from ..config import logging_config

logger = logging_config.setup_logging()

class PromptHandler(PromptEng):

    def __init__(self, question: str, rag: str, assistant: bool = False):        
        self.question = question        
        self.rag = rag
        self.assistant = assistant
        self.role = PromptRole()
        self.rar = PromptRaR()
        self.prompt_response = PromptResponse()
        self.self_verification = PromptSelfVerification()

    def _generatePromptH1(self):
        logger.info("Calling Hypotheses 1")
        system_content = f'{self.role.string_role_complete}' 
        return self.prompt_message(system_content=system_content,
                                   rag_content=self._get_rag(),
                                   user_content=self.question,
                                   role_type= "assistant" if self.assistant else "system")

    def _generatePromptH2(self):
        logger.info("Calling Hypotheses 2")
        system_content = f'{self.role.string_role_complete}' 
        user_content = f'{self.question} \n {self.rar.rar}'
        return self.prompt_message(system_content=system_content,
                                   rag_content=self._get_rag(),
                                   user_content=user_content,
                                   role_type= "assistant" if self.assistant else "system")   

    def _generatePromptH3(self):
        logger.info("Calling Hypotheses 3")
        zcot = PromptZeroCoT()
        system_content = f'{self.role.string_role_complete}' 
        user_content = f'{self.question} \n {zcot.zero_cot_opt1}'
        return self.prompt_message(system_content=system_content,
                                   user_content=user_content,
                                   role_type= "assistant" if self.assistant else "system")

    def _generatePromptH4(self):
        logger.info("Calling Hypotheses 4")
        zcot = PromptZeroCoT()
        system_content = f'{self.role.string_role_complete}' 
        user_content = f'{self.question} \n {zcot.zero_cot_opt1}'
        return self.prompt_message(system_content=system_content,
                                   rag_content=self._get_rag(),
                                   user_content=user_content,
                                   role_type= "assistant" if self.assistant else "system")
    
    def _generatePromptH5(self, **kwargs):
        logger.info("Calling Hypotheses 5")
        thought = kwargs.get("thought", 1)
        system_content = f'{self.role.string_role_complete}'
        skeleton = PromptSkeleton(question=self.question)
        user_content = ""
        rag_content = None
        match thought:
            case 1:
                logger.info(f"Creating thought {thought}")
                user_content = f'{self.question} \n {skeleton.first_think}'                               
            case 2:
                logger.info(f"Creating thought {thought}")
                user_content = f'{self.question} \n {skeleton.second_think}'
            case 3:
                logger.info(f"Creating thought {thought}")
                rag_content = self._get_rag() 
                user_content = f'{self.question} \n {skeleton.third_think}'
        return self.prompt_message(system_content=system_content,
                                   rag_content=rag_content,
                                   user_content=user_content,
                                   role_type= "assistant" if self.assistant else "system")
    
    def _generatePromptH6(self):
        logger.info("Calling Hypotheses 6")
        system_content = f'{self.role.string_role_complete}' 
        user_content = f'{self.question} \n {self.self_verification.self_verification}'        
        return self.prompt_message(system_content=system_content,
                                   rag_content=self._get_rag(),
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
            case "h1":
                return self._generatePromptH1()
            case "h2":
                return self._generatePromptH2()
            case "h3":
                return self._generatePromptH3()
            case "h4":
                return self._generatePromptH4(**kwargs)
            case "h5":
                return self._generatePromptH5()
            case "h6":
                return self._generatePromptH6()
            case _:
                return "Unknown hypotheses choise again."
