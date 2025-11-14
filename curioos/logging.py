"""
Logging Configuration Module

This module sets up centralized logging for CurioOS with file rotation and structured
formatting. It configures both console and file logging with consistent formatting
across the application.

Key features:
- Rotating file handler (prevents unbounded log file growth)
- Consistent timestamp and level formatting
- Avoids duplicate handlers on module reload
- UTF-8 encoding for international character support

Note:
	We import the standard library as 'py_logging' because this module is named
	'logging', which would otherwise shadow the standard library.
"""

from __future__ import annotations

import logging as py_logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def init_logging(log_level: str = "INFO", log_file: Path | None = None) -> None:
	"""
	Initialize application-wide logging with file rotation and formatting.

	This function configures the Python logging system for CurioOS:
	1. Sets the console logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
	2. Creates a rotating file handler to prevent disk space issues
	3. Applies consistent formatting to all log messages
	4. Prevents duplicate handlers if called multiple times

	The rotating file handler keeps up to 2 backup files, with each file
	capped at ~2MB. When the current log exceeds 2MB, it's rotated to
	curioos.log.1, and a new curioos.log is created.

	Args:
		log_level: Logging verbosity as string (default: "INFO")
		           Valid values: DEBUG, INFO, WARNING, ERROR, CRITICAL
		log_file: Optional path for log file (default: ./curioos.log)

	Log Format:
		Each log line follows this pattern:
		YYYY-MM-DD HH:MM:SS | LEVEL | module.name | Message text

	Example:
		2025-01-14 12:30:45 | INFO | curioos.app | Starting CurioOS...
		2025-01-14 12:30:46 | DEBUG | curioos.index.embeddings | Loading model...

	Note:
		Safe to call multiple times - checks for existing RotatingFileHandler
		to avoid duplicate log entries.
	"""
	# Convert string log level to Python logging constant (e.g., "INFO" → logging.INFO)
	# Falls back to INFO if an invalid level is provided
	level = getattr(py_logging, log_level.upper(), py_logging.INFO)
	py_logging.basicConfig(level=level)

	# Default log file location if not specified
	if log_file is None:
		log_file = Path("./curioos.log").resolve()

	# Create rotating file handler:
	# - maxBytes=2,000,000: Rotate when file reaches ~2MB
	# - backupCount=2: Keep curioos.log.1 and curioos.log.2 as backups
	# - encoding="utf-8": Support international characters
	handler = RotatingFileHandler(str(log_file), maxBytes=2_000_000, backupCount=2, encoding="utf-8")

	# Format log messages with timestamp, level, module name, and message
	# Example: "2025-01-14 12:30:45 | INFO | curioos.app | Application started"
	formatter = py_logging.Formatter(
		fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S",
	)
	handler.setFormatter(formatter)

	# Get the root logger (affects all loggers in the application)
	root = py_logging.getLogger()

	# Only add the handler if we don't already have a RotatingFileHandler
	# This prevents duplicate log entries if init_logging is called multiple times
	# (e.g., during testing or module reloads)
	if not any(isinstance(h, RotatingFileHandler) for h in root.handlers):
		root.addHandler(handler)


