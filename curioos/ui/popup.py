from __future__ import annotations

import threading
import time
import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

from pynput import keyboard  # type: ignore


class PopupUI:
	def __init__(self, title: str, hotkey: str, ask_fn: Callable[[str], str]):
		self.title = title
		self.hotkey = hotkey
		self.ask_fn = ask_fn
		
		self._hotkey_listener = None

		self.root = tk.Tk()
		self.root.title(self.title)
		self.root.attributes("-topmost", True)
		self.root.geometry("640x420+200+120")
		self.root.protocol("WM_DELETE_WINDOW", self.root.withdraw)

		self._build_widgets()
		self._bind_hotkey()

	def _build_widgets(self) -> None:
		main = ttk.Frame(self.root, padding=12)
		main.pack(fill="both", expand=True)

		self.query_var = tk.StringVar()
		entry = ttk.Entry(main, textvariable=self.query_var)
		entry.pack(fill="x")
		entry.bind("<Return>", self._on_submit)
		entry.focus_set()

		self.answer = tk.Text(main, height=18, wrap="word")
		self.answer.pack(fill="both", expand=True, pady=8)

		self.status_var = tk.StringVar(value="Ready.")
		status = ttk.Label(main, textvariable=self.status_var, anchor="w")
		status.pack(fill="x")

	def _on_submit(self, event=None) -> None:
		q = self.query_var.get().strip()
		if not q:
			return
		self._run_query(q)

	def _run_query(self, question: str) -> None:
		self.status_var.set("Thinking...")
		self.answer.delete("1.0", tk.END)

		def worker():
			start = time.time()
			try:
				resp = self.ask_fn(question)
				err = None
			except Exception as e:
				resp = ""
				err = str(e)
			elapsed_ms = (time.time() - start) * 1000
	
			def update_ui():
				if err:
					self.answer.insert(tk.END, f"Error: {err}")
				else:
					self.answer.insert(tk.END, resp)
				self.status_var.set(f"Done in {elapsed_ms:.0f} ms")
	
			self.root.after(0, update_ui)

		threading.Thread(target=worker, daemon=True).start()

	def _bind_hotkey(self) -> None:
		hotkey_str = self._format_hotkey_for_pynput(self.hotkey)
		try:
			hot = keyboard.HotKey(keyboard.HotKey.parse(hotkey_str), self._toggle_visibility)
		except Exception:
			# Fallback to a safe default
			hot = keyboard.HotKey(keyboard.HotKey.parse("<ctrl>+<shift>+<space>"), self._toggle_visibility)

		def on_press(key):
			hot.press(key)

		def on_release(key):
			hot.release(key)

		self._hotkey_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
		self._hotkey_listener.daemon = True
		self._hotkey_listener.start()

	def _toggle_visibility(self) -> None:
		self.root.after(0, self._toggle_visibility_main_thread)

	def _toggle_visibility_main_thread(self) -> None:
		if self.root.state() == "withdrawn":
			self.root.deiconify()
			self.root.lift()
			self.root.attributes("-topmost", True)
		else:
			self.root.withdraw()

	def _format_hotkey_for_pynput(self, s: str) -> str:
		# Convert forms like "ctrl+shift+space" → "<ctrl>+<shift>+<space>"
		parts = [p.strip().lower() for p in s.split("+") if p.strip()]
		def needs_brackets(tok: str) -> bool:
			# Single letters/digits can be unbracketed; others bracketed
			if len(tok) == 1 and tok.isalnum():
				return False
			return True
		formatted = "+".join(f"<{p}>" if needs_brackets(p) else p for p in parts)
		return formatted

	def run(self) -> None:
		# Start hidden; show with hotkey
		self.root.withdraw()
		self.root.mainloop()


