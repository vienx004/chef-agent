import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from chef_agent.rag.base import BaseRetriever

logger = logging.getLogger(__name__)

class JSONRecipeRetriever(BaseRetriever):
    """
    A lightweight, keyword-matching recipe retriever that acts as a placeholder
    for a full RAG Vector Database.
    Reads recipes from a local JSON file.
    """

    def __init__(self, db_path: str):
        """
        Initializes the JSON recipe retriever.

        Args:
            db_path (str): Path to the recipes.json database file.
        """
        self.db_path = Path(db_path)
        self.recipes: List[Dict[str, Any]] = []
        self._load_database()

    def _load_database(self) -> None:
        """Loads recipes from the configured JSON file."""
        if not self.db_path.exists():
            logger.warning(
                f"Recipe database file not found at: {self.db_path.resolve()}. "
                "Retriever will return empty results until database is created."
            )
            self.recipes = []
            return

        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                self.recipes = json.load(f)
                logger.info(f"Loaded {len(self.recipes)} recipes from local DB.")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse recipe database JSON: {e}")
            self.recipes = []
        except Exception as e:
            logger.error(f"Unexpected error loading recipe database: {e}")
            self.recipes = []

    def retrieve(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Searches the loaded recipes for keyword matches in title, description, or keywords list.

        Args:
            query (str): The search query.
            limit (int): The maximum number of matches to return.

        Returns:
            List[Dict[str, Any]]: A list of matching recipe dictionaries.
        """
        if not query or not self.recipes:
            return []

        # Standardize query words for scoring
        query_words = [word.lower() for word in query.split() if len(word) > 2]
        if not query_words:
            # Fall back to single characters/words if all search terms are very short
            query_words = [word.lower() for word in query.split() if word]

        matches = []
        for recipe in self.recipes:
            score = 0
            title = recipe.get("title", "").lower()
            description = recipe.get("description", "").lower()
            keywords = [k.lower() for k in recipe.get("keywords", [])]

            for word in query_words:
                # Add score weight depending on where the match is found
                if word in title:
                    score += 5
                if word in keywords:
                    score += 3
                if word in description:
                    score += 1

            if score > 0:
                matches.append((score, recipe))

        # Sort matches by score descending
        matches.sort(key=lambda x: x[0], reverse=True)

        # Extract recipe dicts from sorted list and enforce the limit
        retrieved_recipes = [recipe for _, recipe in matches[:limit]]
        logger.debug(f"Retrieved {len(retrieved_recipes)} matches for query '{query}'")
        return retrieved_recipes

    def add_recipe(self, recipe: Dict[str, Any]) -> None:
        """
        Appends a recipe to the local in-memory database and writes it to recipes.json.

        Args:
            recipe (Dict[str, Any]): Recipe dict containing title, ingredients, etc.
        """
        title = recipe.get("title", "")
        if not title:
            logger.warning("Attempted to add a recipe with no title. Skipping.")
            return

        recipe_id = recipe.get("id") or title.lower().strip().replace(" ", "-")
        recipe["id"] = recipe_id

        # Check if a recipe with this ID or title already exists in the file
        existing_idx = -1
        for idx, r in enumerate(self.recipes):
            if r.get("id") == recipe_id or r.get("title", "").lower().strip() == title.lower().strip():
                existing_idx = idx
                break

        if existing_idx != -1:
            self.recipes[existing_idx] = recipe
            logger.info(f"Updated recipe '{title}' in JSON database.")
        else:
            self.recipes.append(recipe)
            logger.info(f"Added new recipe '{title}' to JSON database.")

        # Save to database file
        try:
            # Ensure the parent directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(self.recipes, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save recipe database to file {self.db_path}: {e}")

