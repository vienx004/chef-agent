import abc
from typing import List, Dict, Optional, Union, Generator

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
