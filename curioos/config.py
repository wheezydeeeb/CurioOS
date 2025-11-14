from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass
class AppConfig:
	"""Runtime configuration for CurioOS."""
	groq_api_key: str
	groq_model: str
	vault_dir: Path
	index_dir: Path
	models_cache_dir: Path
	hotkey: str
	embed_model: str
	log_level: str = "INFO"


def load_config(env_path: Optional[Path] = None) -> AppConfig:
	"""Load configuration from environment and sensible defaults."""
	if env_path is None:
		env_path = Path(".") / ".env"

	load_dotenv(dotenv_path=env_path if env_path.exists() else None)

	vault_dir = Path(os.getenv("CURIO_VAULT", "./data/vault")).resolve()
	index_dir = Path(os.getenv("CURIO_INDEX", "./data/index")).resolve()
	models_cache_dir = Path(os.getenv("CURIO_MODELS_CACHE", "./models")).resolve()

	# Ensure directories exist
	vault_dir.mkdir(parents=True, exist_ok=True)
	index_dir.mkdir(parents=True, exist_ok=True)
	models_cache_dir.mkdir(parents=True, exist_ok=True)

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

	# Handle decommissioned model aliases
	deprecated_models = {
		"llama-3.1-70b-versatile": "llama-3.3-70b-versatile",
	}
	if cfg.groq_model in deprecated_models:
		cfg.groq_model = deprecated_models[cfg.groq_model]

	if not cfg.groq_api_key:
		# Allow running ingestion/index-only flows without API key; UI will warn later
		pass

	return cfg


