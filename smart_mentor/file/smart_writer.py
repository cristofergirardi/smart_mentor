from ..lib.core import WriterFile
from ..config import logging_config
import os
import pandas as pd

logger = logging_config.setup_logging()

class SmartWriter(WriterFile):
    
    def write(self, filename: str, df: pd.DataFrame):        
        if os.path.exists(filename):
            logger.info(f"Writting the file {filename}")
            df.to_csv(filename, mode='a', header=False, index=False)
        else:
            logger.info(f"Creating and writting the file {filename}")
            df.to_csv(filename, mode='w', header=True, index=False)
