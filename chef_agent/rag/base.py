import abc
from typing import List, Dict, Any

class BaseRetriever(abc.ABC):
    """
    Abstract Base Class for retrieval pipelines (RAG).
    
    Implementing classes should connect to search systems or vector databases
    (e.g., ChromaDB, Qdrant, Pinecone, ElasticSearch) and return matching documents.
    """

    @abc.abstractmethod
    def retrieve(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves relevant recipes or culinary notes based on a search query.

        Args:
            query (str): The search query or raw user query.
            limit (int): The maximum number of documents to return. Defaults to 3.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing matched documents.
                Each document dictionary should contain at least:
                - 'title': Title of the document
                - 'description': Summary/description
                - 'ingredients': List of ingredients (optional)
                - 'steps': List of instructions (optional)
        """
        pass
