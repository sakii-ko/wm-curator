"""Tests for Pixi configuration invariants."""

import tomllib
from pathlib import Path


def test_pixi_dev_pypi_dependencies_include_required_cli_imports() -> None:
    """Ensure dev Pixi dependencies include modules imported by the CLI."""
    repo_root = Path(__file__).resolve().parents[1]
    pixi = tomllib.loads((repo_root / "pixi.toml").read_text(encoding="utf-8"))

    assert pixi["feature"]["dev"]["pypi-dependencies"]["fabric"] == "==3.2.2"
