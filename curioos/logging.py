from __future__ import annotations

import logging as py_logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def init_logging(log_level: str = "INFO", log_file: Path | None = None) -> None:
	"""Initialize application logging.

	We keep the module name 'logging' in-package, so import stdlib as py_logging.
	"""
	level = getattr(py_logging, log_level.upper(), py_logging.INFO)
	py_logging.basicConfig(level=level)

	if log_file is None:
		log_file = Path("./curioos.log").resolve()

	handler = RotatingFileHandler(str(log_file), maxBytes=2_000_000, backupCount=2, encoding="utf-8")
	formatter = py_logging.Formatter(
		fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S",
	)
	handler.setFormatter(formatter)

	root = py_logging.getLogger()
	# Avoid duplicate handlers in reloads
	if not any(isinstance(h, RotatingFileHandler) for h in root.handlers):
		root.addHandler(handler)


