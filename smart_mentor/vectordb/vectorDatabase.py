from langchain_core.vectorstores import VectorStoreRetriever
from ..lib.core import ModelAIFactory
from ..config import logging_config
from langchain_chroma import Chroma

logger = logging_config.setup_logging()

class VectorDatabase():
    def __init__(self, persist_dir: str, embedder: ModelAIFactory, documents: list =None):
        if documents:
            logger.info('Creating new vector db')
            self.vectordb = Chroma.from_documents(
                documents=documents,
                embedding=embedder.send_request(),
                persist_directory=persist_dir
            )
        else:
            logger.info('Using existing vector db')
            self.vectordb = Chroma(
                persist_directory=persist_dir,
                embedding_function=embedder.send_request()
            )

    def retriever_data(self, **kwargs):        
        search_kwargs = kwargs.get("search_kwargs","")
        logger.info(f"Retriever data from vectorStore using those parameters {search_kwargs}")
        return self.vectordb.as_retriever(search_kwargs = search_kwargs)
    
    def persist(self):
        logger.info("Persisting vectorStore")
        self.vectordb.persist()

    def invoke(self, vectorRetriver: VectorStoreRetriever, user_question: str ):
        logger.info(f"Invoking vectorStoreRetriver using this parameter {user_question}")
        return vectorRetriver.invoke(user_question)

    def get_metadata(self):
        metadata = self.vectordb.get()["metadatas"]
        return metadata        