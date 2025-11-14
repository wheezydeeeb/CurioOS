from __future__ import annotations

from typing import List

from langchain_groq import ChatGroq  # type: ignore
from langchain_core.messages import SystemMessage, HumanMessage  # type: ignore


class GroqClient:
	"""Thin wrapper using LangChain's ChatGroq for chat completions."""

	def __init__(self, api_key: str, model: str):
		self.api_key = api_key
		self.model = model
		# ChatGroq reads GROQ_API_KEY from env; we keep api_key for validation and potential future use.
		self.llm = ChatGroq(model=self.model) if api_key else None

	def generate(self, messages: List[dict], temperature: float = 0.2, max_tokens: int = 600) -> str:
		if not self.llm:
			return "Groq API key not configured. Please set GROQ_API_KEY in your .env."

		lc_messages = []
		for m in messages:
			role = m.get("role", "user")
			content = m.get("content", "")
			if role == "system":
				lc_messages.append(SystemMessage(content=content))
			else:
				lc_messages.append(HumanMessage(content=content))

		runnable = self.llm.bind(temperature=temperature, max_tokens=max_tokens)
		resp = runnable.invoke(lc_messages)
		return getattr(resp, "content", str(resp)) or ""


