"""
Configuration Management Module

This module handles all application configuration for CurioOS, loading settings from
environment variables and .env files with sensible defaults. It centralizes all
configuration logic to ensure consistency across the application.

Key responsibilities:
- Load environment variables from .env file
- Provide default values for all settings
- Ensure required directories exist
- Handle deprecated model aliases
- Type-safe configuration via Pydantic BaseSettings
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
	"""
	Runtime configuration container for CurioOS.

	This Pydantic settings model loads all application settings from environment variables
	or defaults. All paths are resolved to absolute paths and created automatically.

	Attributes:
		groq_api_key: API key for Groq LLM service (required for question answering)
		groq_model: Name of the Groq model to use (default: llama-3.3-70b-versatile)
		vault_dir: Directory where user documents are stored (default: ./data/vault)
		index_dir: Directory where vector index is persisted (default: ./data/chroma)
		models_cache_dir: Directory for caching embedding models (default: ./models)
		hotkey: Global hotkey for UI popup (default: ctrl+shift+space, currently unused)
		embed_model: Sentence transformer model name (default: all-MiniLM-L6-v2)
		log_level: Logging verbosity level (default: INFO)
	"""
	groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")
	groq_model: str = Field(default="llama-3.3-70b-versatile", validation_alias="GROQ_MODEL")
	vault_dir: Path = Field(default=Path("./data/vault"), validation_alias="CURIO_VAULT")
	index_dir: Path = Field(default=Path("./data/chroma"), validation_alias="CURIO_INDEX")
	models_cache_dir: Path = Field(default=Path("./models"), validation_alias="CURIO_MODELS_CACHE")
	hotkey: str = Field(default="ctrl+shift+space", validation_alias="HOTKEY")
	embed_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", validation_alias="EMBED_MODEL")
	log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

	model_config = SettingsConfigDict(
		env_file=".env",
		env_file_encoding="utf-8",
		case_sensitive=False,
		extra="ignore",
	)

	@field_validator("vault_dir", "index_dir", "models_cache_dir", mode="after")
	@classmethod
	def resolve_and_create_paths(cls, v: Path) -> Path:
		"""Resolve paths to absolute and create directories if they don't exist."""
		resolved = v.resolve()
		resolved.mkdir(parents=True, exist_ok=True)
		return resolved

	@field_validator("groq_model", mode="after")
	@classmethod
	def upgrade_deprecated_models(cls, v: str) -> str:
		"""Handle deprecated/decommissioned Groq model names by auto-upgrading."""
		deprecated_models = {
			"llama-3.1-70b-versatile": "llama-3.3-70b-versatile",
		}
		return deprecated_models.get(v, v)


def load_config(env_path: Optional[Path] = None) -> AppConfig:
	"""
	Load application configuration from environment variables with fallback defaults.

	This function is the primary entry point for configuration. Pydantic BaseSettings
	automatically handles:
	1. Loading variables from .env file (if present)
	2. Reading environment variables with sensible defaults
	3. Creating all required directories (via validators)
	4. Upgrading deprecated model names (via validators)

	Args:
		env_path: Optional path to .env file (default: ./.env)

	Returns:
		AppConfig: Fully initialized configuration object with all paths resolved

	Environment Variables:
		GROQ_API_KEY: Groq API key (default: empty string)
		GROQ_MODEL: LLM model name (default: llama-3.3-70b-versatile)
		CURIO_VAULT: Document storage directory (default: ./data/vault)
		CURIO_INDEX: Vector index storage directory (default: ./data/chroma)
		CURIO_MODELS_CACHE: Embedding model cache directory (default: ./models)
		HOTKEY: Global hotkey for popup UI (default: ctrl+shift+space)
		EMBED_MODEL: Sentence transformer model (default: sentence-transformers/all-MiniLM-L6-v2)
		LOG_LEVEL: Logging level - DEBUG, INFO, WARNING, ERROR (default: INFO)

	Note:
		Missing GROQ_API_KEY is allowed for index-only operations, but will
		prevent question answering functionality.
	"""
	# If a custom .env path is provided, use it; otherwise use default
	if env_path is not None:
		return AppConfig(_env_file=str(env_path))

	# Use default .env file location (./.env)
	return AppConfig()


