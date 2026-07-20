"""Alcove — Conversation Memory.

Persists conversation history to ~/.alcove/memory.json so context survives
app restarts. Thread-safe writes via threading.Lock.
"""

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

ALCOVE_DIR = Path.home() / ".alcove"
MEMORY_FILE = ALCOVE_DIR / "memory.json"
MAX_ENTRIES = 50


class MemoryJournal:
    """Thread-safe conversation memory with persona tagging."""

    def __init__(self, filepath: Path = MEMORY_FILE):
        self._filepath = filepath
        self._lock = threading.Lock()
        self._ensure_dir()

    def _ensure_dir(self):
        """Create ~/.alcove/ and memory file if needed."""
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        if not self._filepath.exists():
            self._filepath.write_text("[]")

    def save_interaction(self, role: str, content: str, persona: str = "") -> None:
        """Save a conversation turn. Thread-safe."""
        with self._lock:
            history = self._read()
            history.append({
                "timestamp": datetime.now().isoformat(),
                "role": role,
                "content": content,
                "persona": persona,
            })
            # Rolling window
            if len(history) > MAX_ENTRIES:
                history = history[-MAX_ENTRIES:]
            self._write(history)

    def load_history(self, limit: int = 50, persona: str = "") -> List[Dict[str, Any]]:
        """Load recent history, optionally filtered by persona."""
        with self._lock:
            history = self._read()[-limit:]
        if persona:
            history = [e for e in history if e.get("persona") == persona]
        return history

    def get_context_string(self, limit: int = 10) -> str:
        """Build a summary string for injection into persona prompts."""
        history = self.load_history(limit)
        if not history:
            return ""
        lines = []
        for msg in history:
            role = msg["role"].upper()
            persona_tag = f" [{msg['persona']}]" if msg.get("persona") else ""
            lines.append(f"{role}{persona_tag}: {msg['content']}")
        return "\n".join(lines)

    def _read(self) -> List[Dict[str, Any]]:
        """Read memory file (caller must hold lock)."""
        try:
            return json.loads(self._filepath.read_text())
        except Exception:
            return []

    def _write(self, data: List[Dict[str, Any]]) -> None:
        """Write memory file (caller must hold lock)."""
        try:
            self._filepath.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Failed to write memory: {e}")
