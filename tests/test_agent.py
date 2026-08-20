import unittest
from typing import List, Dict, Any

from chef_agent.agent import ChefAgent
from chef_agent.rag.base import BaseRetriever
from tests.test_llm import MockLLM

class MockRetriever(BaseRetriever):
    """
    Mock implementation of BaseRetriever for agent testing.
    """
    def __init__(self, stub_results: List[Dict[str, Any]]):
        self.stub_results = stub_results
        self.last_query = None

    def retrieve(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        self.last_query = query
        return self.stub_results[:limit]


class TestChefAgent(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MockLLM(response_text="Delicious recipe instructions.")
        self.stub_recipe = {
            "title": "Stub Souffle",
            "description": "Tasty mock dessert",
            "ingredients": ["chocolate", "flour"],
            "steps": ["Step 1: Whisk", "Step 2: Bake"]
        }
        self.mock_retriever = MockRetriever(stub_results=[self.stub_recipe])
        self.agent = ChefAgent(llm_client=self.mock_llm, retriever=self.mock_retriever)

    def test_agent_synchronous_rag_injection(self):
        """Verifies that the agent retrieves recipes and injects them into the prompt."""
        result = self.agent.add_user_message("Give me a soufflé recipe")
        
        # Verify LLM response
        self.assertEqual(result["response"], "Delicious recipe instructions.")
        self.assertEqual(len(result["retrieved_docs"]), 1)
        self.assertEqual(result["retrieved_docs"][0]["title"], "Stub Souffle")

        # Verify LLM was sent the augmented prompt containing '<retrieved_info>'
        self.assertIn("<retrieved_info>", self.mock_llm.last_prompt)
        self.assertIn("Stub Souffle", self.mock_llm.last_prompt)
        self.assertIn("User Query: Give me a soufflé recipe", self.mock_llm.last_prompt)

        # Verify chat history contains original user message and LLM response (not the augmented prompt)
        self.assertEqual(len(self.agent.history), 2)
        self.assertEqual(self.agent.history[0]["role"], "user")
        self.assertEqual(self.agent.history[0]["content"], "Give me a soufflé recipe")
        self.assertEqual(self.agent.history[1]["role"], "model")
        self.assertEqual(self.agent.history[1]["content"], "Delicious recipe instructions.")

    def test_agent_no_rag_injection_when_no_matches(self):
        """Verifies agent doesn't inject RAG context when retriever has no matches."""
        empty_retriever = MockRetriever(stub_results=[])
        agent = ChefAgent(llm_client=self.mock_llm, retriever=empty_retriever)
        
        result = agent.add_user_message("Make something completely random")
        self.assertEqual(result["retrieved_docs"], [])
        
        # Augmented prompt should simply be the user's message
        self.assertEqual(self.mock_llm.last_prompt, "Make something completely random")

    def test_agent_streaming_rag_injection(self):
        """Verifies streaming output flows correctly and updates history upon completion."""
        stream_generator = self.agent.add_user_message_stream("How do I make chocolate souffle?")
        
        updates = list(stream_generator)
        # Should have at least two updates: the token chunk, and the complete signal
        self.assertGreaterEqual(len(updates), 2)
        
        # First update should yield retrieved documents
        self.assertEqual(updates[0]["retrieved_docs"][0]["title"], "Stub Souffle")
        self.assertEqual(updates[0]["chunk"], "Delicious recipe instructions.")
        
        # Final update should signal completeness and contain full response
        final_update = updates[-1]
        self.assertTrue(final_update["complete"])
        self.assertEqual(final_update["full_response"], "Delicious recipe instructions.")

        # History should be updated at the end
        self.assertEqual(len(self.agent.history), 2)
        self.assertEqual(self.agent.history[1]["content"], "Delicious recipe instructions.")

    def test_clear_history(self):
        """Verifies clearing conversation history resets the history thread."""
        self.agent.add_user_message("Hello Chef!")
        self.assertEqual(len(self.agent.history), 2)
        
        self.agent.clear_history()
        self.assertEqual(len(self.agent.history), 0)

if __name__ == "__main__":
    unittest.main()
