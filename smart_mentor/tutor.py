from .config.config_helper import ConfigHelper
from .orchestrator import SmartMentorOrchestrator
from .config import logging_config


logger = logging_config.setup_logging()

class SmartMentor():

    def __init__(self, config: ConfigHelper):
        super().__init__()
        self.orchestrator = SmartMentorOrchestrator(config)
        logger.info("Smart Tutor is on! \o/")
    
    def get_response(self, question):
        return self.orchestrator.request(question)
    

if __name__ == "__main__":
    config = ConfigHelper()
    tutor = SmartMentor(config)
    prompt = "Escreva Hello Word"   
    print(tutor.get_response(prompt))    
