import unittest
from typing import List, Dict, Optional, Union, Generator

from chef_agent.llm.base import BaseLLM

class MockLLM(BaseLLM):
    """
    Mock implementation of BaseLLM used for local testing without calling APIs.
    """

    def __init__(self, response_text: str = "Chef Response"):
        self.response_text = response_text
        self.last_prompt: Optional[str] = None
        self.last_system_instruction: Optional[str] = None
        self.last_history: Optional[List[Dict[str, str]]] = None

    def generate_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        stream: bool = False
    ) -> Union[str, Generator[str, None, None]]:
        self.last_prompt = prompt
        self.last_system_instruction = system_instruction
        self.last_history = history

        if stream:
            def generator() -> Generator[str, None, None]:
                yield self.response_text
            return generator()
        return self.response_text


class TestLLMInterface(unittest.TestCase):
    def test_mock_llm_instantiation(self):
        """Verifies that subclasses of BaseLLM can be instantiated and fulfill the interface contract."""
        mock = MockLLM(response_text="Hello cooking fan!")
        self.assertTrue(isinstance(mock, BaseLLM))
        
        # Test synchronous call
        res = mock.generate_response(prompt="How to fry an egg?")
        self.assertEqual(res, "Hello cooking fan!")
        self.assertEqual(mock.last_prompt, "How to fry an egg?")
        
        # Test streaming call
        stream_res = mock.generate_response(prompt="How to bake bread?", stream=True)
        chunks = list(stream_res)
        self.assertEqual(chunks, ["Hello cooking fan!"])
        self.assertEqual(mock.last_prompt, "How to bake bread?")

if __name__ == "__main__":
    unittest.main()
