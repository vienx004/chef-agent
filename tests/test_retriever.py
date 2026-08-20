import unittest
import tempfile
import json
import os
from pathlib import Path

from chef_agent.rag.retriever import JSONRecipeRetriever

class TestJSONRecipeRetriever(unittest.TestCase):
    def setUp(self):
        # Create a temporary JSON database file for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_recipes.json"
        
        self.sample_data = [
            {
                "id": "recipe-1",
                "title": "Gluten-Free Chocolate Cake",
                "description": "A delicious rich chocolate cake without gluten.",
                "ingredients": ["chocolate", "eggs", "almond flour"],
                "steps": ["Melt chocolate", "Mix everything", "Bake"],
                "keywords": ["chocolate", "cake", "dessert", "gluten-free"]
            },
            {
                "id": "recipe-2",
                "title": "Classic Pancake Batter",
                "description": "Fluffy pancakes best served with maple syrup.",
                "ingredients": ["flour", "milk", "eggs", "buttermilk"],
                "steps": ["Whisk wet ingredients", "Fold in dry", "Cook on griddle"],
                "keywords": ["pancake", "breakfast", "buttermilk"]
            }
        ]
        
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self.sample_data, f)
            
        self.retriever = JSONRecipeRetriever(db_path=str(self.db_path))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_database(self):
        """Verifies database is loaded properly."""
        self.assertEqual(len(self.retriever.recipes), 2)
        self.assertEqual(self.retriever.recipes[0]["id"], "recipe-1")

    def test_retrieve_exact_match(self):
        """Verifies exact matches against title, description, or keywords return correctly."""
        results = self.retriever.retrieve("chocolate")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "recipe-1")

    def test_retrieve_relevance_scoring(self):
        """Verifies sorting matches based on weight scores (title match > description match)."""
        # "chocolate" is in title of recipe-1 (score: 5).
        # "buttermilk" is in keywords of recipe-2 (score: 3).
        # Searching "chocolate buttermilk" matches both, but recipe-1 should be ranked higher.
        results = self.retriever.retrieve("chocolate buttermilk", limit=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], "recipe-1")
        self.assertEqual(results[1]["id"], "recipe-2")

    def test_retrieve_limit(self):
        """Verifies limit parameter limits results."""
        results = self.retriever.retrieve("chocolate pancake", limit=1)
        self.assertEqual(len(results), 1)

    def test_retrieve_empty_query(self):
        """Verifies empty queries return an empty list."""
        results = self.retriever.retrieve("")
        self.assertEqual(results, [])

    def test_nonexistent_file(self):
        """Verifies retriever gracefully handles missing files."""
        bad_retriever = JSONRecipeRetriever("nonexistent_file.json")
        self.assertEqual(bad_retriever.recipes, [])
        self.assertEqual(bad_retriever.retrieve("chocolate"), [])
        
if __name__ == "__main__":
    unittest.main()
