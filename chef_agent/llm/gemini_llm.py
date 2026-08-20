import os
from typing import List, Dict, Optional, Union, Generator, Any
from google import genai
from google.genai import types
from google.genai.errors import APIError

from chef_agent.llm.base import BaseLLM

class GeminiLLM(BaseLLM):
    """
    Gemini client implementation using the new google-genai SDK.
    Provides concrete implementations for generating synchronous and streaming responses.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-3.6-flash"):
        """
        Initializes the Gemini client.

        Args:
            api_key (str, optional): The Gemini API key. If not provided, will look for the 
                GEMINI_API_KEY environment variable.
            model_name (str): Model name string. Defaults to "gemini-3.6-flash".
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name

        if not self.api_key:
            raise ValueError(
                "Gemini API key is missing. Please set GEMINI_API_KEY in your .env file or environment."
            )

        # Initialize the official google-genai Client
        self.client = genai.Client(api_key=self.api_key)

    def _convert_history(self, history: List[Dict[str, str]], current_prompt: str) -> List[types.Content]:
        """
        Helper method to map a generic list of history dicts:
            [{'role': 'user'|'model'|'assistant', 'content': '...'}]
        into standard google-genai types.Content objects.

        Args:
            history: List of chat history structures.
            current_prompt: The new user prompt that triggers this generation.

        Returns:
            List[types.Content]: Formatted conversation history compatible with the API contents arg.
        """
        contents = []

        # Convert historical turns
        for turn in history:
            role = turn.get("role", "user")
            content_text = turn.get("content", "")

            # Gemini expects 'user' or 'model' roles.
            # We map 'assistant' and 'chef' to 'model' for compatibility.
            gemini_role = "model" if role in ["model", "assistant", "chef"] else "user"

            contents.append(
                types.Content(
                    role=gemini_role,
                    parts=[types.Part.from_text(text=content_text)]
                )
            )

        # Append the new user prompt as the final turn
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=current_prompt)]
            )
        )

        return contents

    def generate_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        stream: bool = False
    ) -> Union[str, Generator[str, None, None]]:
        """
        Calls the Gemini API to get a response.

        Args:
            prompt: User message prompt.
            system_instruction: Chef instructions defining persona.
            history: Thread history list.
            stream: Boolean flag to enable/disable token streaming.

        Returns:
            Union[str, Generator[str, None, None]]: Full string response or a generator of string chunks.
        """
        # 1. Transform history into Google Content objects
        contents = self._convert_history(history or [], prompt)

        # 2. Build the Generation config
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.7,
        )

        try:
            if stream:
                # Request a stream of chunks from the API
                response_stream = self.client.models.generate_content_stream(
                    model=self.model_name,
                    contents=contents,
                    config=config
                )

                def generator() -> Generator[str, None, None]:
                    for chunk in response_stream:
                        if chunk.text:
                            yield chunk.text

                return generator()
            else:
                # Request a synchronous single response
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config
                )
                return response.text or ""

        except APIError as e:
            # Provide high-fidelity exception wrapping
            raise RuntimeError(f"Gemini API request failed: {e.message} (status_code: {e.code})") from e
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred during LLM generation: {str(e)}") from e
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates vector embeddings for a list of input texts using Gemini's gemini-embedding-001.

        Args:
            texts (List[str]): List of strings to be vectorized.

        Returns:
            List[List[float]]: List of vector embeddings (lists of floats).
        """
        if not texts:
            return []

        try:
            # Generate embeddings using the official SDK call
            response = self.client.models.embed_content(
                model="gemini-embedding-001",
                contents=texts
            )
            return [emb.values for emb in response.embeddings]
        except APIError as e:
            raise RuntimeError(f"Gemini embedding API call failed: {e.message} (status_code: {e.code})") from e
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred during embedding generation: {str(e)}") from e

    def extract_structured_data(self, text: str, schema_class: type) -> Optional[Any]:
        """
        Analyzes unstructured text and extracts structured fields matching a Pydantic BaseModel schema.
        Uses Gemini's built-in structured JSON output capability.

        Args:
            text (str): The raw text data.
            schema_class (type): The Pydantic model class defining the desired schema structure.

        Returns:
            Optional[Any]: An instance of schema_class filled with extracted data, or None if extraction fails.
        """
        if not text:
            return None

        # Build structured generation config forcing json format matching schema_class
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema_class,
            temperature=0.0  # Keep it deterministic for high-fidelity extraction
        )

        try:
            # We call the model with a system context prompt instructing it to extract the information
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=f"Analyze the following conversation block and extract the relevant data structured exactly to match the schema:\n\n{text}",
                config=config
            )
            # The parsed field is automatically populated by the SDK with the Pydantic instance
            return response.parsed
        except Exception as e:
            logger.error(f"Structured data extraction failed: {str(e)}")
            return None

