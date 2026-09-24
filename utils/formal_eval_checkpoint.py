"""Resolve preferred formal-evaluation checkpoints with a latest fallback."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)")


def _absolute_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def is_complete_model_checkpoint(path: Path) -> bool:
    """Return whether a directory contains config plus loadable model weights."""
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    direct_weights = (
        path / "model.safetensors",
        path / "pytorch_model.bin",
    )
    if any(candidate.is_file() for candidate in direct_weights):
        return True
    sharded_indexes = (
        path / "model.safetensors.index.json",
        path / "pytorch_model.bin.index.json",
    )
    return any(candidate.is_file() for candidate in sharded_indexes)


def resolve_checkpoint_spec(
    spec: str | Path | dict[str, Any],
    repo_root: Path,
) -> tuple[Path, dict[str, Any] | None]:
    """Resolve a literal path or ``preferred`` plus ``fallback_latest_under``."""
    if isinstance(spec, (str, Path)):
        return _absolute_path(spec, repo_root), None
    if not isinstance(spec, dict):
        raise TypeError(
            "checkpoint must be a path string or a mapping with "
            "preferred/fallback_latest_under."
        )

    allowed_keys = {"preferred", "fallback_latest_under"}
    unknown_keys = sorted(set(spec) - allowed_keys)
    if unknown_keys:
        raise ValueError(f"Unknown checkpoint selection keys: {unknown_keys}")
    if "preferred" not in spec or "fallback_latest_under" not in spec:
        raise ValueError(
            "Checkpoint selection requires both preferred and "
            "fallback_latest_under."
        )

    preferred = _absolute_path(spec["preferred"], repo_root)
    fallback_root = _absolute_path(spec["fallback_latest_under"], repo_root)
    if is_complete_model_checkpoint(preferred):
        return preferred, {
            "preferred": str(preferred),
            "fallback_latest_under": str(fallback_root),
            "resolved": str(preferred),
            "used_fallback": False,
        }

    candidates: list[tuple[int, Path]] = []
    if fallback_root.is_dir():
        for candidate in fallback_root.iterdir():
            match = CHECKPOINT_PATTERN.fullmatch(candidate.name)
            if match and is_complete_model_checkpoint(candidate):
                candidates.append((int(match.group(1)), candidate))
    if not candidates:
        raise FileNotFoundError(
            f"Preferred checkpoint is incomplete or missing ({preferred}), "
            f"and no complete checkpoint-* exists under {fallback_root}."
        )
    resolved = max(candidates, key=lambda item: item[0])[1]
    return resolved, {
        "preferred": str(preferred),
        "fallback_latest_under": str(fallback_root),
        "resolved": str(resolved),
        "used_fallback": True,
    }


def resolve_method_checkpoints(
    config: dict[str, Any],
    repo_root: Path,
) -> None:
    """Resolve method checkpoint specs in place before validation/aggregation."""
    methods = config.get("methods", {})
    if not isinstance(methods, dict):
        raise ValueError("config.methods must be a mapping.")
    for method_name, method in methods.items():
        if not isinstance(method, dict):
            raise ValueError(f"Method {method_name!r} must be a mapping.")
        if "checkpoint" not in method:
            raise ValueError(f"Method {method_name!r} has no checkpoint.")
        resolved, selection = resolve_checkpoint_spec(
            method["checkpoint"],
            repo_root,
        )
        if selection is not None:
            method["checkpoint"] = str(resolved)
            method["checkpoint_selection"] = selection
