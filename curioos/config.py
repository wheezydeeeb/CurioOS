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
- Type-safe configuration via dataclass
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass
class AppConfig:
	"""
	Runtime configuration container for CurioOS.

	This dataclass holds all application settings, loaded from environment variables
	or defaults. All paths are resolved to absolute paths to avoid ambiguity.

	Attributes:
		groq_api_key: API key for Groq LLM service (required for question answering)
		groq_model: Name of the Groq model to use (default: llama-3.3-70b-versatile)
		vault_dir: Directory where user documents are stored (default: ./data/vault)
		index_dir: Directory where vector index is persisted (default: ./data/index)
		models_cache_dir: Directory for caching embedding models (default: ./models)
		hotkey: Global hotkey for UI popup (default: ctrl+shift+space, currently unused)
		embed_model: Sentence transformer model name (default: all-MiniLM-L6-v2)
		log_level: Logging verbosity level (default: INFO)
	"""
	groq_api_key: str
	groq_model: str
	vault_dir: Path
	index_dir: Path
	models_cache_dir: Path
	hotkey: str
	embed_model: str
	log_level: str = "INFO"


def load_config(env_path: Optional[Path] = None) -> AppConfig:
	"""
	Load application configuration from environment variables with fallback defaults.

	This function is the primary entry point for configuration. It:
	1. Loads variables from .env file (if present)
	2. Reads environment variables with sensible defaults
	3. Creates all required directories automatically
	4. Handles deprecated model names by auto-upgrading them

	Args:
		env_path: Optional path to .env file (default: ./.env)

	Returns:
		AppConfig: Fully initialized configuration object with all paths resolved

	Environment Variables:
		GROQ_API_KEY: Groq API key (default: empty string)
		GROQ_MODEL: LLM model name (default: llama-3.3-70b-versatile)
		CURIO_VAULT: Document storage directory (default: ./data/vault)
		CURIO_INDEX: Vector index storage directory (default: ./data/index)
		CURIO_MODELS_CACHE: Embedding model cache directory (default: ./models)
		HOTKEY: Global hotkey for popup UI (default: ctrl+shift+space)
		EMBED_MODEL: Sentence transformer model (default: sentence-transformers/all-MiniLM-L6-v2)
		LOG_LEVEL: Logging level - DEBUG, INFO, WARNING, ERROR (default: INFO)

	Note:
		Missing GROQ_API_KEY is allowed for index-only operations, but will
		prevent question answering functionality.
	"""
	# Determine which .env file to use (defaults to ./.env)
	if env_path is None:
		env_path = Path(".") / ".env"

	# Load environment variables from .env file if it exists
	# If file doesn't exist, python-dotenv silently continues (no error)
	load_dotenv(dotenv_path=env_path if env_path.exists() else None)

	# Load directory paths from environment with defaults, then resolve to absolute paths
	# resolve() ensures we have full paths regardless of current working directory
	vault_dir = Path(os.getenv("CURIO_VAULT", "./data/vault")).resolve()
	index_dir = Path(os.getenv("CURIO_INDEX", "./data/index")).resolve()
	models_cache_dir = Path(os.getenv("CURIO_MODELS_CACHE", "./models")).resolve()

	# Create all required directories if they don't exist
	# parents=True creates intermediate directories, exist_ok=True prevents errors if already exists
	vault_dir.mkdir(parents=True, exist_ok=True)
	index_dir.mkdir(parents=True, exist_ok=True)
	models_cache_dir.mkdir(parents=True, exist_ok=True)

	# Build configuration object from environment variables or defaults
	cfg = AppConfig(
		groq_api_key=os.getenv("GROQ_API_KEY", ""),
		groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
		vault_dir=vault_dir,
		index_dir=index_dir,
		models_cache_dir=models_cache_dir,
		hotkey=os.getenv("HOTKEY", "ctrl+shift+space"),
		embed_model=os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
		log_level=os.getenv("LOG_LEVEL", "INFO"),
	)

	# Handle deprecated/decommissioned Groq model names by auto-upgrading to current versions
	# This ensures backward compatibility when Groq deprecates old model names
	deprecated_models = {
		"llama-3.1-70b-versatile": "llama-3.3-70b-versatile",
	}
	if cfg.groq_model in deprecated_models:
		cfg.groq_model = deprecated_models[cfg.groq_model]

	# Missing API key is acceptable - allows index building without LLM access
	# The app will warn the user later if they try to ask questions without a key
	if not cfg.groq_api_key:
		pass  # Explicitly showing we're okay with empty API key for now

	return cfg


