from ..models_api.openai.embedding_openAi import EmbeddingOpenAi
from ..vectordb.vectorDatabase import VectorDatabase
from ..config import logging_config
import pandas as pd

logger = logging_config.setup_logging()

class RAG:    
    def __init__(self, embedding_config: dict, name_deployment: str, persist_dir: str, top_k: int):        
        self.embeddingOpenAi = EmbeddingOpenAi(embedding_config, name_deployment = name_deployment)
        self.vectorDB = VectorDatabase(persist_dir = persist_dir, embedder = self.embeddingOpenAi)
        self.TOP_K = top_k
    
    def retrieve(self, question: str, list_filter:list = None):
        if list_filter:
            logger.info("Using filters")
            for filter in list_filter:
                select_filter = {'type': {'$eq': filter}}
                retriever = self.vectorDB.retriever_data(search_kwargs={'k': self.TOP_K, 
                                                                        'filter': select_filter})
        else:
            logger.info('Filters not being used')
            retriever = self.vectorDB.retriever_data(search_kwargs={'k': self.TOP_K})

        docs = self.vectorDB.invoke(retriever, question)
        docs = [chunk for chunk in docs]

        return docs
    
    def retrieve_metadata(self):        
        df_metadatas = pd.DataFrame(self.vectorDB.get_metadata())
        unique_values = pd.unique(df_metadatas.values.ravel())
        return list(unique_values)
    
    def save_documents(self, question: str = None, answer: str = None):
        # it is purpose create two if statements to prevent to add None value in the vectordb
        if question:
            logger.info("Adding a question in the vectordb")
            self.vectorDB.add_document(text=question, metadata={"type": "question"})

        # it is purpose create two if statements to prevent to add None value in the vectordb
        if answer:
            logger.info("Adding a answer in the vectordb")
            self.vectorDB.add_document(text=answer, metadata={"type": "answer"})