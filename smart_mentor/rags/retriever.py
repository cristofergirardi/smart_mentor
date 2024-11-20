from ..modelos_api.openai.embedding_openAi import EmbeddingOpenAi
from ..vectordb.vectorDatabase import VectorDatabase
from ..config import logging_config
import numpy as np

logger = logging_config.setup_logging()

class RAG:    
    def __init__(self, embedding_config, name_deployment, persist_dir, top_k):        
        self.embeddingOpenAi = EmbeddingOpenAi(embedding_config, name_deployment = name_deployment)
        self.vectorDB = VectorDatabase(persist_dir = persist_dir, embedder = self.embeddingOpenAi)
        self.TOP_K = top_k
    
    def retrieve(self, question):
        retriever = self.vectorDB.retriever_data(search_kwargs={'k': self.TOP_K})

        docs = self.vectorDB.invoke(retriever, question)
        docs = [chunk for chunk in docs]

        return docs
    
    def retrieve_modules(self):
        metadata = self.vectorDB.get_metadata()
        modules = [m['module'] for m in metadata]
        modules = list(np.unique(modules))
        logger.info(f"Modules: {modules}")
        return modules
    