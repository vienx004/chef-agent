import abc
from typing import List, Dict, Optional, Union, Generator, Any

class BaseLLM(abc.ABC):
    """
    Abstract Base Class for LLM Client integrations.
    
    Any new LLM provider (Gemini, OpenAI, Anthropic, local Llama, etc.) should implement
    this interface to ensure it can be seamlessly swapped in the Chef Agent.
    """

    @abc.abstractmethod
    def generate_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        stream: bool = False
    ) -> Union[str, Generator[str, None, None]]:
        """
        Generates a response from the LLM based on user input, system instructions, and chat history.

        Args:
            prompt (str): The current user query or instructions.
            system_instruction (str, optional): The system prompt/personality instruction. Defaults to None.
            history (list, optional): List of prior chat turn dicts, e.g.
                [{'role': 'user', 'content': 'hello'}, {'role': 'model', 'content': 'hi'}]
            stream (bool): Whether to stream the response chunks or return the full string at once.

        Returns:
            Union[str, Generator[str, None, None]]: The string response, or a generator yielding response chunks.
        """
        pass

    @abc.abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates vector embeddings for a list of input texts.

        Args:
            texts (List[str]): List of texts to generate vector embeddings for.

        Returns:
            List[List[float]]: List of float vector embeddings corresponding to the input texts.
        """
        pass

    @abc.abstractmethod
    def extract_structured_data(self, text: str, schema_class: type) -> Optional[Any]:
        """
        Analyzes unstructured text and extracts structured fields matching a Pydantic BaseModel schema.

        Args:
            text (str): The raw text data.
            schema_class (type): The Pydantic model class defining the desired schema structure.

        Returns:
            Optional[Any]: An instance of schema_class filled with extracted data, or None if extraction fails.
        """
        pass


