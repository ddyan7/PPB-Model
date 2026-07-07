"""Shared utilities: project paths, config loading, seeding, and logging.

These helpers keep the rest of the library free of hard-coded absolute paths and
ensure every stage is reproducible from a fixed seed and a single config file.
"""
from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def project_root() -> Path:
    """Return the project root directory (two levels above this file: src/ppb_model/)."""
    return Path(__file__).resolve().parents[2]


def resolve_path(path_like: str | os.PathLike[str]) -> Path:
    """Resolve a possibly-relative path against the project root.

    Absolute paths are returned unchanged; relative paths are anchored at the
    project root so scripts work regardless of the current working directory.
    """
    p = Path(path_like)
    return p if p.is_absolute() else (project_root() / p)


def load_config(config_path: str | os.PathLike[str] = "configs/default.yaml") -> dict[str, Any]:
    """Load a YAML config file into a plain dict.

    Raises:
        FileNotFoundError: if the config file does not exist.
        ValueError: if the file does not parse to a mapping.
    """
    cfg_path = resolve_path(config_path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config did not parse to a mapping: {cfg_path}")
    return cfg


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and hash randomisation for reproducibility."""
    if not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed).__name__}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_logger(name: str, log_file: str | os.PathLike[str] | None = None,
               level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger that writes to stdout and optionally a file.

    Idempotent: repeated calls with the same name do not duplicate handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    if log_file is not None:
        log_path = resolve_path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


@dataclass(frozen=True)
class Paths:
    """Canonical project directories, resolved from the project root."""

    root: Path
    raw: Path
    interim: Path
    processed: Path
    splits: Path
    figures: Path
    tables: Path
    results: Path
    models: Path
    logs: Path

    @classmethod
    def create(cls) -> "Paths":
        r = project_root()
        paths = cls(
            root=r,
            raw=r / "data" / "raw",
            interim=r / "data" / "interim",
            processed=r / "data" / "processed",
            splits=r / "data" / "splits",
            figures=r / "reports" / "figures",
            tables=r / "reports" / "tables",
            results=r / "reports" / "results",
            models=r / "models",
            logs=r / "logs",
        )
        for d in (paths.interim, paths.processed, paths.splits, paths.figures,
                  paths.tables, paths.results, paths.models, paths.logs):
            d.mkdir(parents=True, exist_ok=True)
        return paths
