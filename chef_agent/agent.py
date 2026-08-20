import logging
from typing import List, Dict, Optional, Union, Generator, Any
from pydantic import BaseModel, Field

from chef_agent.config import CHEF_SYSTEM_INSTRUCTION
from chef_agent.llm.base import BaseLLM
from chef_agent.rag.base import BaseRetriever

logger = logging.getLogger(__name__)

class RecipeExtractionSchema(BaseModel):
    """
    Pydantic schema to extract structured recipe data from the model's chat output.
    """
    is_recipe: bool = Field(description="True if the text contains a complete recipe with a title, ingredients, and instructions. False otherwise.")
    title: Optional[str] = Field(None, description="The name/title of the recipe.")
    description: Optional[str] = Field(None, description="A brief mouth-watering summary description of the dish.")
    ingredients: Optional[List[str]] = Field(None, description="List of ingredients with measurements.")
    steps: Optional[List[str]] = Field(None, description="List of cooking instructions/steps.")
    keywords: Optional[List[str]] = Field(None, description="Relevant tags (e.g. baking, dessert, chicken, dinner).")

class ChefAgent:
    """
    Coordinator class for the Chef AI Agent.
    Manages the conversation state, retrieves relevant recipe context using RAG,
    constructs the final LLM prompt, and coordinates with the pluggable LLM provider.
    """

    def __init__(self, llm_client: BaseLLM, retriever: Optional[BaseRetriever] = None):
        """
        Initializes the Chef Agent with an LLM backend and an optional RAG retriever.

        Args:
            llm_client (BaseLLM): An instantiated subclass of BaseLLM.
            retriever (BaseRetriever, optional): An instantiated subclass of BaseRetriever.
        """
        self.llm = llm_client
        self.retriever = retriever
        
        # Thread history stored as a list of dictionaries with 'role' and 'content' keys.
        # Format: [{'role': 'user', 'content': '...'}, {'role': 'model', 'content': '...'}]
        self.history: List[Dict[str, str]] = []

    def clear_history(self) -> None:
        """Resets the conversation history."""
        self.history.clear()
        logger.info("Chef Agent conversation history cleared.")

    def _prepare_augmented_prompt(self, user_message: str) -> tuple[str, Optional[List[Dict[str, Any]]]]:
        """
        Helper method to search the RAG retriever and inject findings into the user message.

        Args:
            user_message (str): The raw input message from the user.

        Returns:
            tuple[str, Optional[List[Dict[str, Any]]]]: A tuple containing:
                - The augmented prompt string (with <retrieved_info> block if any records found).
                - The list of retrieved documents (or None if none were matched).
        """
        retrieved_docs = None
        augmented_prompt = user_message

        # If a retriever is active, attempt to fetch matching culinary knowledge
        if self.retriever:
            try:
                retrieved_docs = self.retriever.retrieve(user_message, limit=2)
                
                if retrieved_docs:
                    # Construct the context block to inject into the LLM context
                    context_block = "<retrieved_info>\n"
                    context_block += "The following verified culinary reference data was retrieved from the database:\n\n"
                    
                    for idx, doc in enumerate(retrieved_docs, 1):
                        context_block += f"--- Document {idx}: {doc.get('title', 'Untitled')} ---\n"
                        context_block += f"Description: {doc.get('description', '')}\n"
                        
                        if "ingredients" in doc:
                            context_block += "Ingredients:\n"
                            for ing in doc["ingredients"]:
                                context_block += f"  - {ing}\n"
                                
                        if "steps" in doc:
                            context_block += "Method:\n"
                            for step in doc["steps"]:
                                context_block += f"  - {step}\n"
                        context_block += "\n"
                        
                    context_block += "</retrieved_info>\n"
                    
                    # Augment the current prompt with the context block
                    augmented_prompt = f"{context_block}\nUser Query: {user_message}"
                    logger.info(f"Augmented prompt with {len(retrieved_docs)} document(s).")
            except Exception as e:
                # Log retrieve failure but do not break the chat flow
                logger.error(f"Failed to retrieve documents from RAG: {e}", exc_info=True)

        return augmented_prompt, retrieved_docs

    def add_user_message(self, user_message: str) -> Dict[str, Any]:
        """
        Processes a user message, performs RAG retrieval, invokes the LLM synchronously,
        and saves both the request and response in history.

        Args:
            user_message (str): The message from the user.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - 'response': The response text from the Chef.
                - 'retrieved_docs': List of documents injected for context (if any).
        """
        # 1. Retrieve RAG context and augment prompt
        augmented_prompt, retrieved_docs = self._prepare_augmented_prompt(user_message)

        # 2. Invoke the pluggable LLM client
        response_text = self.llm.generate_response(
            prompt=augmented_prompt,
            system_instruction=CHEF_SYSTEM_INSTRUCTION,
            history=self.history,
            stream=False
        )
        
        # Ensure we type-check the output
        if not isinstance(response_text, str):
            response_text = str(response_text)

        # 3. Save original message and final response to history
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "model", "content": response_text})

        # 4. Check if LLM response contains a new recipe, and save to database if so
        saved_recipe = self._check_and_save_recipe(response_text)

        return {
            "response": response_text,
            "retrieved_docs": retrieved_docs,
            "saved_recipe": saved_recipe
        }

    def add_user_message_stream(self, user_message: str) -> Generator[Dict[str, Any], None, None]:
        """
        Processes a user message and returns a generator yielding response chunks as they arrive.
        Saves the final aggregated response to history once finished.

        Args:
            user_message (str): The message from the user.

        Yields:
            Dict[str, Any]: Dictionaries containing:
                - 'chunk': Individual text token/chunk.
                - 'retrieved_docs': Injected documents (yielded in the first chunk, otherwise None).
                - 'complete': Boolean indicating if streaming is finished.
                - 'full_response': The final concatenated string (yielded only when complete is True).
        """
        # 1. Retrieve RAG context and augment prompt
        augmented_prompt, retrieved_docs = self._prepare_augmented_prompt(user_message)

        # 2. Request a streaming response from the pluggable LLM
        response_generator = self.llm.generate_response(
            prompt=augmented_prompt,
            system_instruction=CHEF_SYSTEM_INSTRUCTION,
            history=self.history,
            stream=True
        )

        full_response = ""
        is_first_chunk = True

        for chunk in response_generator:
            full_response += chunk
            yield {
                "chunk": chunk,
                "retrieved_docs": retrieved_docs if is_first_chunk else None,
                "complete": False
            }
            is_first_chunk = False

        # 3. Store conversation history upon completion
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "model", "content": full_response})

        # 4. Check if LLM response contains a new recipe, and save to database if so
        saved_recipe = self._check_and_save_recipe(full_response)

        yield {
            "chunk": "",
            "retrieved_docs": None,
            "complete": True,
            "full_response": full_response,
            "saved_recipe": saved_recipe
        }

    def _check_and_save_recipe(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Helper method to analyze the Chef's response, extract structured recipes if present,
        and save them to the configured retriever.
        """
        if not self.retriever:
            return None

        try:
            # Call structured extraction on the LLM client
            extracted = self.llm.extract_structured_data(response_text, RecipeExtractionSchema)
            
            if extracted and extracted.is_recipe and extracted.title and extracted.ingredients and extracted.steps:
                # Convert Pydantic object to database-compatible dictionary
                recipe_dict = {
                    "id": extracted.title.lower().strip().replace(" ", "-"),
                    "title": extracted.title,
                    "description": extracted.description or "",
                    "ingredients": extracted.ingredients,
                    "steps": extracted.steps,
                    "keywords": extracted.keywords or []
                }
                
                # Insert into local JSON file or ChromaDB
                self.retriever.add_recipe(recipe_dict)
                return recipe_dict
        except Exception as e:
            logger.error(f"Failed to auto-extract or save recipe: {e}", exc_info=True)
            
        return None
