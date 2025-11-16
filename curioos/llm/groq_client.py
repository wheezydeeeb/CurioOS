"""
Groq LLM Client Module

This module provides a thin wrapper around Groq's LLM API via LangChain's ChatGroq.
Groq provides ultra-fast inference for open-source models like Llama, Mixtral, and Gemma.

Key features:
- Simple message-based interface (system + user messages)
- Configurable temperature and token limits
- Graceful handling of missing API keys
- Consistent string output

Supported Models (as of 2025):
	- llama-3.3-70b-versatile (default): Fast, high-quality general-purpose model
	- llama-3.1-8b-instant: Smallest/fastest Llama variant
	- mixtral-8x7b-32768: Large context window (32k tokens)
	- gemma-7b-it: Google's Gemma instruction-tuned model

Design Philosophy:
	Keep it simple. This wrapper exists to:
	1. Provide sensible defaults for RAG use case
	2. Gracefully handle missing API keys (allows index-only operation)
	3. Accept LangChain message types directly for type safety

Typical Usage:
	>>> from langchain_core.messages import SystemMessage, HumanMessage
	>>> client = GroqClient(api_key="gsk_...", model="llama-3.3-70b-versatile")
	>>> messages = [
	...     SystemMessage(content="You are a helpful assistant."),
	...     HumanMessage(content="What is CurioOS?")
	... ]
	>>> answer = client.generate(messages, temperature=0.2, max_tokens=600)
"""

from __future__ import annotations

from typing import List
import asyncio

from langchain_groq import ChatGroq  # type: ignore
from langchain_core.messages import AnyMessage  # type: ignore


class GroqClient:
	"""
	Thin wrapper around LangChain's ChatGroq for chat completions.

	This class provides a simplified interface to Groq's LLM API, handling
	message formatting and providing sensible defaults for RAG applications.
	"""

	def __init__(self, api_key: str, model: str):
		"""
		Initialize the Groq client.

		Args:
			api_key: Groq API key (get from https://console.groq.com)
			model: Model name (e.g., "llama-3.3-70b-versatile")

		Note:
			If api_key is empty, self.llm will be None and generate() will
			return an error message. This allows the app to run in index-only
			mode without API credentials.

		Environment Variables:
			ChatGroq reads GROQ_API_KEY from environment, so we don't need
			to pass it explicitly. We store api_key here for validation only.
		"""
		self.api_key = api_key
		self.model = model

		# Create LangChain ChatGroq instance only if we have an API key
		# Pass api_key explicitly to ChatGroq (it also reads from GROQ_API_KEY env var)
		self.llm = ChatGroq(model=self.model, api_key=api_key) if api_key else None

	def generate(self, messages: List[AnyMessage], temperature: float = 0.2, max_tokens: int = 600) -> str:
		"""
		Generate a completion from the LLM given a list of LangChain messages.

		Args:
			messages: List of LangChain message objects (SystemMessage, HumanMessage, AIMessage, etc.)
			          Example: [SystemMessage(content="..."), HumanMessage(content="...")]
			temperature: Sampling temperature (0.0 = deterministic, 2.0 = very random)
			             Lower values (0.0-0.5) are better for factual/analytical tasks
			             Higher values (0.7-2.0) are better for creative tasks
			max_tokens: Maximum number of tokens to generate (limits response length)

		Returns:
			Generated text as string
			If API key is missing, returns an error message

		Temperature Guidance for RAG:
			- 0.0-0.3: Factual Q&A, precise answers (recommended for CurioOS)
			- 0.4-0.7: Balanced creativity and accuracy
			- 0.8-1.2: Creative writing, brainstorming
			- 1.3+: Experimental, very creative (often incoherent)

		Max Tokens Guidance:
			- 100-300: Short, concise answers
			- 300-600: Medium-length answers (default for CurioOS)
			- 600-1000: Detailed explanations
			- 1000+: Long-form content

		Example:
			>>> from langchain_core.messages import SystemMessage, HumanMessage
			>>> messages = [
			...     SystemMessage(content="Answer concisely."),
			...     HumanMessage(content="What is Python?")
			... ]
			>>> answer = client.generate(messages, temperature=0.2, max_tokens=200)
			>>> print(answer)
			Python is a high-level programming language known for its simplicity...
		"""
		# Check if API key is configured
		if not self.llm:
			return "Groq API key not configured. Please set GROQ_API_KEY in your .env."

		# Bind generation parameters to the LLM
		# This creates a "runnable" with fixed temperature and max_tokens
		runnable = self.llm.bind(temperature=temperature, max_tokens=max_tokens)

		# Invoke the LLM with the messages
		# Returns an AIMessage object with the generated content
		resp = runnable.invoke(messages)

		# Extract content from response
		# getattr with fallback ensures we always return a string
		return getattr(resp, "content", str(resp)) or ""

	async def agenerate(self, messages: List[AnyMessage], temperature: float = 0.2, max_tokens: int = 600) -> str:
		"""
		Async version of generate() for concurrent LLM calls.

		This method allows multiple LLM generations to run concurrently,
		improving throughput when processing multiple queries.

		Args:
			messages: List of LangChain message objects
			temperature: Sampling temperature (0.0 = deterministic, 2.0 = very random)
			max_tokens: Maximum number of tokens to generate

		Returns:
			Generated text as string

		Example:
			>>> async def process_questions(questions):
			...     tasks = [client.agenerate([HumanMessage(content=q)]) for q in questions]
			...     answers = await asyncio.gather(*tasks)
			...     return answers
		"""
		# Check if API key is configured
		if not self.llm:
			return "Groq API key not configured. Please set GROQ_API_KEY in your .env."

		# Bind generation parameters to the LLM
		runnable = self.llm.bind(temperature=temperature, max_tokens=max_tokens)

		# Async invoke the LLM with the messages
		resp = await runnable.ainvoke(messages)

		# Extract content from response
		return getattr(resp, "content", str(resp)) or ""
