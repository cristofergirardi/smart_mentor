"""core"""
from abc import ABC, abstractmethod
import pandas as pd

class ModelAIFactory(ABC):
    @abstractmethod
    def send_request(self, **kwargs): # pragma: no cover
        pass

class ReaderFile(ABC):
    @abstractmethod
    def checkFile(self, filename: str): # pragma: no cover
        pass
    
    @abstractmethod
    def readFile(self, filename:str): # pragma: no cover
        pass

    @abstractmethod
    def checkFile(self, filename:str): # pragma: no cover
        pass

class WriterFile(ABC):
    @abstractmethod
    def write(self, filename:str, **kwargs): # pragma: no cover
        pass
    

class PromptEng(ABC):
    @abstractmethod
    def generatePrompt(self): # pragma: no cover
        pass

    @abstractmethod
    def prompt_message(sefl): # pragma: no cover
        pass