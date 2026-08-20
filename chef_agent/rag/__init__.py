"""
RAG (Retrieval-Augmented Generation) Module

Provides abstractions and concrete implementations for searching and retrieving
relevant culinary documents, recipes, and tips to inject into the Chef's context.
"""
from chef_agent.rag.base import BaseRetriever
from chef_agent.rag.retriever import JSONRecipeRetriever

__all__ = ["BaseRetriever", "JSONRecipeRetriever"]
