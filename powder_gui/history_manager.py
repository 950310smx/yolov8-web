"""
History persistence utilities for the powder detection GUI.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


class HistoryManager:
    """Simple JSON-backed history store."""

    def __init__(self, history_path: Path, limit: int = 200) -> None:
        self.path = history_path
        self.limit = limit
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: List[Dict] = self._load()

    def _load(self) -> List[Dict]:
        if not self.path.exists():
            return []
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
        return []

    @property
    def entries(self) -> List[Dict]:
        return list(self._entries)

    def add_entry(self, entry: Dict) -> None:
        self._entries.insert(0, entry)
        if len(self._entries) > self.limit:
            self._entries = self._entries[: self.limit]
        self._save()

    def clear(self) -> None:
        self._entries.clear()
        self._save()

    def _save(self) -> None:
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self._entries, f, indent=2, ensure_ascii=False)


