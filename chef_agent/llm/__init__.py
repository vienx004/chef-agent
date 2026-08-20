"""
LLM Integration Layer

Provides an abstract interface and concrete implementations for calling different
Language Model providers (such as Gemini, OpenAI, etc.).
"""
from chef_agent.llm.base import BaseLLM
from chef_agent.llm.gemini_llm import GeminiLLM

__all__ = ["BaseLLM", "GeminiLLM"]
