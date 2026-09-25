"""YAML-backed configuration for the analysis stage.

Every path lives in one file so the scripts carry no site-specific defaults;
see ``config.example.yaml`` next to the run scripts.
"""

from __future__ import annotations

from pathlib import Path

import yaml


class Config:
    """Thin wrapper over the parsed YAML with friendly errors."""

    def __init__(self, data: dict, source: Path):
        self.data = data
        self.source = source
        #: Relative paths in the YAML are resolved against the config's own
        #: directory, so a config keeps working from any working directory.
        self.base = source.parent

    def path(self, value) -> Path | None:
        """Resolve one configured path against the config file's directory."""
        if value in (None, "", "null"):
            return None
        candidate = Path(str(value)).expanduser()
        return candidate if candidate.is_absolute() else (self.base / candidate)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path)
        if not path.exists():
            raise SystemExit(
                f"config file not found: {path}\n"
                f"Copy config.example.yaml to {path.name} and fill in your paths."
            )
        with open(path) as fh:
            return cls(yaml.safe_load(fh) or {}, path)

    def family(self, name: str, cohort: str) -> dict:
        """Return the ``features:`` block for e.g. ``family('fe3', 'sus')``."""
        try:
            entry = self.data["features"][name][cohort]
        except KeyError as exc:
            raise SystemExit(
                f"{self.source}: missing features.{name}.{cohort} ({exc})"
            ) from exc
        if entry.get("path") in (None, "", "null"):
            raise SystemExit(f"{self.source}: features.{name}.{cohort}.path is empty")

        entry = dict(entry)
        entry["path"] = self.path(entry["path"])
        if entry.get("aggregation_paths"):
            entry["aggregation_paths"] = {
                k: self.path(v) for k, v in entry["aggregation_paths"].items()
            }
        return entry

    def get(self, *keys, default=None):
        node = self.data
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    @property
    def out_dir(self) -> Path:
        out = self.path(self.get("out_dir", default="results"))
        out.mkdir(parents=True, exist_ok=True)
        return out
