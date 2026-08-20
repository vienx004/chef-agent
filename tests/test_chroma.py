import unittest
import tempfile
import json
from pathlib import Path
import chromadb

from chef_agent.rag.chroma_retriever import ChromaRecipeRetriever
from tests.test_llm import MockLLM


class ChromaMockLLM(MockLLM):
    """
    Mock LLM subclass providing mock embeddings aligned with search terms.
    """

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            # Let's create a 3-dimensional mock vector
            vec = [0.0, 0.0, 0.0]
            text_lower = text.lower()
            if "mushroom" in text_lower or "risotto" in text_lower:
                vec[0] = 1.0
            if "lemon" in text_lower or "tart" in text_lower:
                vec[1] = 1.0
            embeddings.append(vec)
        return embeddings


class TestChromaRecipeRetriever(unittest.TestCase):
    def setUp(self):
        # Setup temporary directory only for seed JSON file (no DB files on disk)
        self.temp_file_dir = tempfile.TemporaryDirectory()
        self.seed_path = Path(self.temp_file_dir.name) / "seed_recipes.json"

        self.sample_recipes = [
            {
                "id": "recipe-a",
                "title": "Creamy Mushroom Risotto",
                "description": "Rich Italian arborio rice dish cooked with mushrooms.",
                "ingredients": ["arborio rice", "mushrooms", "parmesan cheese", "chicken stock"],
                "steps": ["Sauté mushrooms", "Add rice and stir", "Slowly add stock"],
                "keywords": ["mushroom", "risotto", "rice", "italian"]
            },
            {
                "id": "recipe-b",
                "title": "Lemon Tart",
                "description": "Sweet and tangy lemon curd in a buttery crust.",
                "ingredients": ["lemon juice", "sugar", "eggs", "flour", "butter"],
                "steps": ["Bake pastry shell", "Cook lemon curd", "Assemble and chill"],
                "keywords": ["lemon", "tart", "dessert", "sweet"]
            }
        ]

        with open(self.seed_path, "w", encoding="utf-8") as f:
            json.dump(self.sample_recipes, f)

        self.mock_llm = ChromaMockLLM()
        
        # Instantiate in-memory Chroma EphemeralClient for lightning fast, lock-free tests
        self.chroma_client = chromadb.EphemeralClient()
        
        self.retriever = ChromaRecipeRetriever(
            llm=self.mock_llm,
            client=self.chroma_client,
            seed_json_path=str(self.seed_path)
        )

    def tearDown(self):
        self.temp_file_dir.cleanup()

    def test_seeding_works(self):
        """Verifies that ChromaDB is successfully seeded with documents on instantiation."""
        # Ensure our collection has 2 records
        self.assertEqual(self.retriever.collection.count(), 2)

    def test_retrieve_format_and_similarity(self):
        """Verifies retrieved format matches JSONRecipeRetriever structure and list unpacking works."""
        results = self.retriever.retrieve("lemon", limit=1)
        self.assertEqual(len(results), 1)
        recipe = results[0]

        # Verify key schema structure
        self.assertEqual(recipe["id"], "recipe-b")
        self.assertEqual(recipe["title"], "Lemon Tart")
        self.assertEqual(recipe["description"], "Sweet and tangy lemon curd in a buttery crust.")
        self.assertEqual(recipe["ingredients"], ["lemon juice", "sugar", "eggs", "flour", "butter"])
        self.assertEqual(recipe["steps"], ["Bake pastry shell", "Cook lemon curd", "Assemble and chill"])
        self.assertEqual(recipe["keywords"], ["lemon", "tart", "dessert", "sweet"])

    def test_retrieve_empty_query(self):
        """Verifies empty queries yield empty results."""
        self.assertEqual(self.retriever.retrieve(""), [])

    def test_retrieve_limit(self):
        """Verifies limit parameter controls result count."""
        results = self.retriever.retrieve("risotto curds", limit=1)
        self.assertEqual(len(results), 1)



if __name__ == "__main__":
    unittest.main()
