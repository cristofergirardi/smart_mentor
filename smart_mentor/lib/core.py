"""core"""
from abc import ABC, abstractmethod
import pandas as pd

class ModelAIFactory(ABC):
    @abstractmethod
    def send_request(self, **kwargs): # pragma: no cover
        pass

