from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Set

from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent, FileDeletedEvent  # type: ignore
from watchdog.observers import Observer  # type: ignore


class _VaultEventHandler(FileSystemEventHandler):
	def __init__(self, on_change: Callable[[Path, str], None], allowed_exts: Set[str]):
		super().__init__()
		self.on_change = on_change
		self.allowed_exts = allowed_exts

	def _handle(self, src_path: str, kind: str) -> None:
		path = Path(src_path)
		if path.suffix.lower() in self.allowed_exts:
			self.on_change(path, kind)

	def on_created(self, event: FileCreatedEvent) -> None:
		self._handle(event.src_path, "created")

	def on_modified(self, event: FileModifiedEvent) -> None:
		self._handle(event.src_path, "modified")

	def on_deleted(self, event: FileDeletedEvent) -> None:
		self._handle(event.src_path, "deleted")


class VaultWatcher:
	"""Watch a folder and invoke callbacks on file changes with simple debounce."""

	def __init__(self, vault_dir: Path, on_change: Callable[[Path, str], None], debounce_sec: float = 1.0):
		self.vault_dir = vault_dir
		self.on_change = on_change
		self.debounce_sec = debounce_sec
		self.observer: Optional[Observer] = None
		self._pending: Dict[Path, str] = {}
		self._lock = threading.Lock()
		self._stop = threading.Event()

	def _accumulate(self, path: Path, kind: str) -> None:
		with self._lock:
			self._pending[path] = kind

	def _worker(self) -> None:
		while not self._stop.is_set():
			time.sleep(self.debounce_sec)
			with self._lock:
				items = list(self._pending.items())
				self._pending.clear()
			for path, kind in items:
				self.on_change(path, kind)

	def start(self) -> None:
		handler = _VaultEventHandler(self._accumulate, {".txt", ".md", ".pdf"})
		self.observer = Observer()
		self.observer.schedule(handler, str(self.vault_dir), recursive=True)
		self.observer.start()
		threading.Thread(target=self._worker, daemon=True).start()

	def stop(self) -> None:
		self._stop.set()
		if self.observer:
			self.observer.stop()
			self.observer.join(timeout=2)


