"""
File System Watcher Module

This module provides real-time monitoring of the document vault directory using
the watchdog library. It detects file changes (create, modify, delete) and triggers
re-indexing automatically with intelligent debouncing to avoid redundant processing.

Key features:
- Monitors vault directory recursively (including subdirectories)
- Filters events to supported file types only (.txt, .md, .pdf)
- Debounces rapid changes (e.g., multiple saves) into single re-index
- Thread-safe event accumulation
- Graceful shutdown

Design Rationale:
	Text editors often save files multiple times in quick succession (autosave,
	user save, etc.). Without debouncing, each save would trigger a full re-index,
	wasting CPU and API calls. The debouncer accumulates changes over a time window
	(default 1 second) and processes them in a batch.

Typical Usage:
	>>> watcher = VaultWatcher(vault_dir, on_file_changed, debounce_sec=1.0)
	>>> watcher.start()
	>>> # ... watcher runs in background ...
	>>> watcher.stop()  # Clean shutdown
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Set

from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent, FileDeletedEvent  # type: ignore
from watchdog.observers import Observer  # type: ignore


class _VaultEventHandler(FileSystemEventHandler):
	"""
	Internal event handler for watchdog that filters and forwards file events.

	This handler receives all file system events from watchdog and filters
	them to only process supported file types. It forwards qualifying events
	to the user-provided callback.
	"""

	def __init__(self, on_change: Callable[[Path, str], None], allowed_exts: Set[str]):
		"""
		Initialize the event handler with callback and file type filter.

		Args:
			on_change: Callback function that receives (Path, event_type)
			allowed_exts: Set of file extensions to monitor (e.g., {".txt", ".md", ".pdf"})
		"""
		super().__init__()
		self.on_change = on_change
		self.allowed_exts = allowed_exts

	def _handle(self, src_path: str, kind: str) -> None:
		"""
		Filter events by file extension and forward to callback.

		Args:
			src_path: Absolute path to the file that changed (as string)
			kind: Event type - "created", "modified", or "deleted"
		"""
		path = Path(src_path)
		# Only process files with allowed extensions
		if path.suffix.lower() in self.allowed_exts:
			self.on_change(path, kind)

	def on_created(self, event: FileCreatedEvent) -> None:
		"""Handle file creation events."""
		self._handle(event.src_path, "created")

	def on_modified(self, event: FileModifiedEvent) -> None:
		"""Handle file modification events."""
		self._handle(event.src_path, "modified")

	def on_deleted(self, event: FileDeletedEvent) -> None:
		"""Handle file deletion events."""
		self._handle(event.src_path, "deleted")


class VaultWatcher:
	"""
	Watch a directory for file changes and invoke callbacks with debouncing.

	This class monitors a directory tree for changes to document files and
	triggers re-indexing after a debounce period. Multiple rapid changes to
	the same file are batched into a single callback invocation.

	Threading Model:
		- Watchdog observer runs in background thread (managed by watchdog library)
		- Debounce worker thread processes accumulated events periodically
		- Lock protects the shared pending events dictionary

	Debouncing Strategy:
		Events are accumulated in a dictionary (path → event_type). If the same
		file changes multiple times within the debounce window, only the latest
		event type is kept. After debounce_sec seconds, all pending events are
		processed together.
	"""

	def __init__(self, vault_dir: Path, on_change: Callable[[Path, str], None], debounce_sec: float = 1.0):
		"""
		Initialize the vault watcher.

		Args:
			vault_dir: Directory to monitor (monitored recursively)
			on_change: Callback function called for each file change
			           Receives (file_path: Path, event_type: str)
			           event_type is one of: "created", "modified", "deleted"
			debounce_sec: Time window in seconds to batch changes (default: 1.0)
			              Higher values reduce processing frequency but increase latency
		"""
		self.vault_dir = vault_dir
		self.on_change = on_change
		self.debounce_sec = debounce_sec
		self.observer: Optional[Observer] = None
		self._pending: Dict[Path, str] = {}  # Maps file paths to latest event type
		self._lock = threading.Lock()  # Protects _pending dictionary
		self._stop = threading.Event()  # Signals worker thread to terminate

	def _accumulate(self, path: Path, kind: str) -> None:
		"""
		Accumulate a file change event for batched processing.

		This method is called by the watchdog event handler whenever a
		relevant file changes. It adds the event to the pending queue.

		Args:
			path: Path to the file that changed
			kind: Event type - "created", "modified", or "deleted"

		Thread Safety:
			Protected by self._lock to prevent concurrent modification
			of the _pending dictionary.
		"""
		with self._lock:
			# If the same file has multiple events, keep only the latest
			# Example: file modified twice → only one "modified" callback
			self._pending[path] = kind

	def _worker(self) -> None:
		"""
		Background worker thread that processes accumulated events periodically.

		This thread runs in a loop, sleeping for debounce_sec between iterations.
		On each iteration, it:
		1. Grabs all pending events (atomically clearing the queue)
		2. Calls the user callback for each event

		The thread terminates when self._stop is set (via stop() method).

		Note:
			Runs as daemon thread so it won't prevent program exit if stop()
			isn't called explicitly.
		"""
		while not self._stop.is_set():
			# Sleep for the debounce period
			time.sleep(self.debounce_sec)

			# Atomically grab and clear pending events
			with self._lock:
				items = list(self._pending.items())
				self._pending.clear()

			# Process each accumulated event outside the lock
			# This allows new events to accumulate while we're processing
			for path, kind in items:
				self.on_change(path, kind)

	def start(self) -> None:
		"""
		Start monitoring the vault directory for changes.

		This method:
		1. Creates a watchdog observer with our event handler
		2. Configures it to monitor vault_dir recursively
		3. Starts the watchdog observer thread
		4. Starts the debounce worker thread

		File Type Filter:
			Only .txt, .md, and .pdf files trigger events

		Thread Safety:
			Safe to call once. Calling multiple times without stop() may
			create duplicate watchers.
		"""
		# Create event handler that filters to supported file types
		handler = _VaultEventHandler(self._accumulate, {".txt", ".md", ".pdf"})

		# Create and configure watchdog observer
		self.observer = Observer()
		self.observer.schedule(handler, str(self.vault_dir), recursive=True)
		self.observer.start()

		# Start debounce worker thread as daemon (won't block program exit)
		threading.Thread(target=self._worker, daemon=True).start()

	def stop(self) -> None:
		"""
		Stop monitoring and clean up resources.

		This method:
		1. Signals the worker thread to stop
		2. Stops the watchdog observer
		3. Waits up to 2 seconds for observer to finish

		After calling stop(), the watcher can be restarted with start().

		Note:
			Pending events may be lost if stop() is called before they're processed.
		"""
		# Signal worker thread to exit its loop
		self._stop.set()

		# Stop watchdog observer
		if self.observer:
			self.observer.stop()
			# Wait for observer thread to finish (timeout after 2 seconds)
			self.observer.join(timeout=2)
