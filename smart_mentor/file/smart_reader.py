import os
import pandas as pd
from pathlib import Path
from ..lib.core import ReaderFile
from ..config import logging_config

logger = logging_config.setup_logging()

class SmartReader(ReaderFile):
    
    def readFile(self, filename: str):
        # Get file and extension        
        file = Path(filename)
        file_extension = file.name
        extension = file.suffix

        if 'xlsx' in extension:
            logger.info(f'Reading {file_extension} file...')
            return pd.read_excel(filename)
        elif 'csv' in extension:
            logger.info(f'Reading {file_extension} file...')
            return pd.read_csv(filename)
        else:
            logger.info(f'Reading {file_extension} file ...')
            with open(filename, "r") as f:
                content = f.read()
                f.close()
            return content
        
    def checkFile(self, filename: str):
        return os.path.exists(filename)
    
    def removeFile(self, filename: str):
        os.remove(filename)
        logger.info(f'Remove {filename}')